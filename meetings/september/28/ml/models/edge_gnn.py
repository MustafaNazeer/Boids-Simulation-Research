import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, MessagePassing

from ml.data import features, labels
from ml.models.encoder import build_encoder


class NodeRefine(MessagePassing):
    """One round of message passing over the candidate graph."""

    def __init__(self, hidden):
        super().__init__(aggr="mean")
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, h, edge_index):
        return h + self.propagate(edge_index, h=h)

    def message(self, h_i, h_j):
        return self.mlp(torch.cat([h_i, h_j], dim=-1))


class TwoRoundRefine(nn.Module):
    """Two rounds of message passing, each with its own weights.

    Separate weights rather than the same layer applied twice, because the
    question is whether a second round of a different transformation helps,
    not whether iterating one transformation helps.
    """

    def __init__(self, hidden):
        super().__init__()
        self.first = NodeRefine(hidden)
        self.second = NodeRefine(hidden)

    def forward(self, h, edge_index):
        return self.second(self.first(h, edge_index), edge_index)


class AttentionRefine(nn.Module):
    """One round of graph attention, in place of mean aggregation.

    NodeRefine aggregates neighbours with an unweighted mean. This learns how
    much to weight each neighbour instead. Heads are averaged rather than
    concatenated so the output width matches every other variant and the rest
    of the model is untouched.

    The residual mirrors NodeRefine, which returns h plus its messages, so the
    two differ only in how neighbours are combined.
    """

    def __init__(self, hidden, heads=4):
        super().__init__()
        self.conv = GATConv(hidden, hidden, heads=heads, concat=False,
                            add_self_loops=False)

    def forward(self, h, edge_index):
        return h + self.conv(h, edge_index)


class NoRefine(nn.Module):
    """No message passing at all. A control, not a candidate.

    Node embeddings pass through untouched, so the pair head scores each pair
    from the two agents' own motion and the edge attributes alone. If this
    matches mp1, the graph half of the model is contributing nothing on this
    task, which bears directly on the project title and is the most important
    thing this sweep could find.
    """

    def forward(self, h, edge_index):
        return h


def build_relational(kind, hidden):
    """One uniform call for every relational variant.

    Every variant has the signature forward(h, edge_index) -> h, same shape in
    and out, so EdgeClassifier does not need to know which one it was given.
    """
    if kind == "mp1":
        return NodeRefine(hidden)
    if kind == "mp2":
        return TwoRoundRefine(hidden)
    if kind == "gat":
        return AttentionRefine(hidden)
    if kind == "none":
        return NoRefine()
    raise ValueError(
        "unknown relational kind %r, expected one of mp1, mp2, gat, none"
        % (kind,))


class EdgeClassifier(nn.Module):
    """GRU first, then GNN, then a symmetric edge head.

    The head is symmetric by construction: it consumes the sum and the
    absolute difference of the two node embeddings, both of which are
    unchanged when the pair order swaps. The labels are undirected, so the
    scores must be too.
    """

    n_relations = len(labels.RELATIONS)

    def __init__(self, hidden=64, edge_dim=2, use_distance=True,
                 in_dim=None, encoder_kind="gru", relational_kind="mp1",
                 k_window=10):
        super().__init__()
        self.use_distance = use_distance
        encoder_in = features.FEATURE_DIM if in_dim is None else in_dim
        self.encoder = build_encoder(encoder_kind, in_dim=encoder_in,
                                     hidden=hidden, k_window=k_window)
        self.refine = build_relational(relational_kind, hidden)
        head_in = 2 * hidden + (edge_dim if use_distance else edge_dim - 1)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.n_relations),
        )

    @classmethod
    def from_arm(cls, arm, in_dim, use_distance=False):
        """Build the configuration an Arm describes.

        `use_distance` defaults to False rather than to the class default,
        because every arm in this experiment is position free. Defaulting the
        other way would silently measure the 0.999 ceiling instead of the
        model, which is the exact failure this whole experiment exists to
        avoid.
        """
        return cls(hidden=arm.hidden, use_distance=use_distance,
                   in_dim=in_dim, encoder_kind=arm.encoder,
                   relational_kind=arm.relational, k_window=arm.k_window)

    def forward(self, x, edge_index, pairs, edge_attr):
        h = self.refine(self.encoder(x), edge_index)
        hi, hj = h[pairs[:, 0]], h[pairs[:, 1]]
        # column 0 of edge_attr is distance, column 1 is relative speed.
        # the ablation drops distance so the model must recover connectivity
        # from motion correlation alone
        attr = edge_attr if self.use_distance else edge_attr[:, 1:]
        return self.head(torch.cat([hi + hj, (hi - hj).abs(), attr], dim=-1))
