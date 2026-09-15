import numpy as np
import torch
import torch.nn as nn

from ml import metrics
from ml.data import labels


def train(model, dataset, epochs, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    model.train()
    for _ in range(epochs):
        total, count = 0.0, 0
        for sample in dataset:
            optimizer.zero_grad()
            logits = model(sample.x, sample.edge_index, sample.pairs,
                           sample.edge_attr)
            loss = loss_fn(logits, sample.y)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            count += 1
        history.append(total / max(count, 1))
    return history


def evaluate(model, dataset, only_special=False):
    """Score a model over a dataset.

    With `only_special=True` the metrics are computed over just those pairs
    touching a leader or preplanned agent. About 4.5 percent of agents are
    special, so an aggregate over all pairs is dominated by ordinary ones and
    cannot show an effect that exists only on the rest. Returns None when no
    such pairs exist, rather than reporting a number computed on nothing.
    """
    model.eval()
    scores, targets = [], []
    with torch.no_grad():
        for sample in dataset:
            logits = model(sample.x, sample.edge_index, sample.pairs,
                           sample.edge_attr)
            s, t = logits.numpy(), sample.y.numpy()
            if only_special:
                keep = sample.special.numpy().astype(bool)
                s, t = s[keep], t[keep]
            if s.shape[0]:
                scores.append(s)
                targets.append(t)

    if not scores:
        return None

    scores = np.concatenate(scores, axis=0)
    targets = np.concatenate(targets, axis=0)

    report = {}
    for col, relation in enumerate(labels.RELATIONS):
        f1, _ = metrics.best_f1(scores[:, col], targets[:, col])
        report[relation] = {
            "auc": metrics.roc_auc(scores[:, col], targets[:, col]),
            "f1": f1,
        }
    return report
