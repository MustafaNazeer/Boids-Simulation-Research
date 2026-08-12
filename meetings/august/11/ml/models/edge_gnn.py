import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from ml.data import labels
from ml.models.encoder import GRUEncoder


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


class EdgeClassifier(nn.Module):
    """GRU first, then GNN, then a symmetric edge head.

    The head is symmetric by construction: it consumes the sum and the
    absolute difference of the two node embeddings, both of which are
    unchanged when the pair order swaps. The labels are undirected, so the
    scores must be too.
    """

    n_relations = len(labels.RELATIONS)

    def __init__(self, hidden=64, edge_dim=2, use_distance=True,
                 in_dim=None):
        super().__init__()
        self.use_distance = use_distance
        self.encoder = (GRUEncoder(hidden=hidden) if in_dim is None
                        else GRUEncoder(in_dim=in_dim, hidden=hidden))
        self.refine = NodeRefine(hidden)
        head_in = 2 * hidden + (edge_dim if use_distance else edge_dim - 1)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.n_relations),
        )

    def forward(self, x, edge_index, pairs, edge_attr):
        h = self.refine(self.encoder(x), edge_index)
        hi, hj = h[pairs[:, 0]], h[pairs[:, 1]]
        # column 0 of edge_attr is distance, column 1 is relative speed.
        # the ablation drops distance so the model must recover connectivity
        # from motion correlation alone
        attr = edge_attr if self.use_distance else edge_attr[:, 1:]
        return self.head(torch.cat([hi + hj, (hi - hj).abs(), attr], dim=-1))
