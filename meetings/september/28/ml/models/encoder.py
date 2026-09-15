import torch
import torch.nn as nn

from ml.data import features


class GRUEncoder(nn.Module):
    """One GRU, weights shared across every agent.

    Agents are carried in the batch dimension, so the parameter count does not
    depend on how many boids a run has. That is what lets a model trained on
    40 boid runs be applied to 160 boid runs.
    """

    def __init__(self, in_dim=features.FEATURE_DIM, hidden=64, layers=1):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True)

    def forward(self, x):
        _, h = self.gru(x)
        return h[-1]


class LSTMEncoder(nn.Module):
    """The same shape as GRUEncoder, with an LSTM instead.

    Three gates and a separate cell state, so about a third more parameters
    than a GRU at the same hidden size. That cost is the thing being measured.
    """

    def __init__(self, in_dim, hidden, layers=1):
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers,
                            batch_first=True)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]


class MeanPoolEncoder(nn.Module):
    """No recurrence and no frame ordering: average the window, then project.

    A control, not a candidate. Averaging over time discards the order of the
    frames entirely, so this arm losing to the GRU does not by itself show that
    recurrence matters. Read it beside MLPEncoder, which keeps the ordering.
    """

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.hidden = hidden
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x):
        return self.proj(x.mean(dim=1))


class MLPEncoder(nn.Module):
    """No recurrence, ordering preserved: flatten the window, then project.

    The sharper of the two non recurrent controls. It sees exactly what the
    GRU sees, in order, and differs only by having no recurrent connection, so
    a tie with the GRU is direct evidence that recurrence is not earning its
    place at this window length.

    Flattening fixes the input width at k_window * in_dim, which is why this
    variant needs the window length at construction time.
    """

    def __init__(self, in_dim, hidden, k_window):
        super().__init__()
        self.hidden = hidden
        self.k_window = int(k_window)
        self.proj = nn.Sequential(
            nn.Linear(self.k_window * in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x):
        if x.shape[1] != self.k_window:
            raise ValueError(
                "MLPEncoder built for a %d frame window, got %d"
                % (self.k_window, x.shape[1]))
        return self.proj(x.reshape(x.shape[0], -1))


class TransformerEncoder(nn.Module):
    """One self attention layer over the window, then average over time.

    Attention is order invariant on its own, so a learned positional embedding
    is added. Without it this would silently be a more expensive meanpool,
    which is the trap worth avoiding in a comparison whose whole point is
    telling architectures apart.
    """

    def __init__(self, in_dim, hidden, k_window, heads=4):
        super().__init__()
        self.hidden = hidden
        self.k_window = int(k_window)
        self.input_proj = nn.Linear(in_dim, hidden)
        self.positions = nn.Parameter(torch.zeros(1, self.k_window, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=2 * hidden,
            batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x):
        h = self.input_proj(x) + self.positions[:, :x.shape[1], :]
        return self.encoder(h).mean(dim=1)


def build_encoder(kind, in_dim, hidden, k_window):
    """One uniform call for every time encoder variant.

    Every variant maps (agents, K, in_dim) to (agents, hidden), with agents
    carried in the batch dimension. `k_window` is ignored by the variants that
    do not need it, so that callers do not have to know which those are.
    """
    if kind == "gru":
        return GRUEncoder(in_dim=in_dim, hidden=hidden)
    if kind == "lstm":
        return LSTMEncoder(in_dim=in_dim, hidden=hidden)
    if kind == "meanpool":
        return MeanPoolEncoder(in_dim=in_dim, hidden=hidden)
    if kind == "mlp":
        return MLPEncoder(in_dim=in_dim, hidden=hidden, k_window=k_window)
    if kind == "transformer":
        return TransformerEncoder(in_dim=in_dim, hidden=hidden,
                                  k_window=k_window)
    raise ValueError(
        "unknown encoder kind %r, expected one of gru, lstm, meanpool, mlp, "
        "transformer" % (kind,))
