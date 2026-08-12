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
