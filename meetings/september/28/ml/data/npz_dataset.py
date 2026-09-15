import os
from dataclasses import dataclass

import numpy as np
import torch

from ml.data import candidates as cand_mod
from ml.data import features as feat_mod
from ml.data import labels as label_mod


@dataclass
class Sample:
    x: torch.Tensor
    edge_index: torch.Tensor
    pairs: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor
    run_id: str
    # True where a pair touches at least one leader or preplanned agent.
    # Only about 4.5 percent of agents are special, so an aggregate metric is
    # dominated by ordinary pairs; this is what lets a caller evaluate on the
    # subset where agent type could possibly matter.
    special: torch.Tensor


class BoidsEdgeDataset:

    def __init__(self, run_dir, k_window=10, k_neighbors=None, stride=10,
                 runs=None, include_position=True, agent_types=None):
        self.run_dir = run_dir
        self.k_window = int(k_window)
        # None means score every pair, which is the measured right default.
        # see candidates.candidate_pairs for the numbers behind that
        self.k_neighbors = None if k_neighbors is None else int(k_neighbors)
        self.stride = int(stride)
        self.include_position = bool(include_position)
        if agent_types not in (None, "type", "noise"):
            raise ValueError("agent_types must be None, 'type' or 'noise', "
                             "got %r" % (agent_types,))
        self.agent_types = agent_types

        available = sorted(f[:-4] for f in os.listdir(run_dir)
                           if f.endswith(".npz"))
        self.runs = available if runs is None else [r for r in available
                                                    if r in set(runs)]
        self._cache = {}
        self._index = []
        for run_id in self.runs:
            frames = self._load(run_id)["positions"].shape[0]
            for t in range(self.k_window - 1, frames, self.stride):
                self._index.append((run_id, t))

    def _load(self, run_id):
        if run_id not in self._cache:
            with np.load(os.path.join(self.run_dir, run_id + ".npz")) as z:
                self._cache[run_id] = {
                    "positions": z["positions"],
                    "velocities": z["velocities"],
                    "leader_mask": np.asarray(z["leader_mask"], dtype=bool),
                    "preplanned_mask": np.asarray(z["preplanned_mask"],
                                                  dtype=bool),
                    "radii": {r: float(z[label_mod.radius_key(r)])
                              for r in label_mod.RELATIONS},
                }
        return self._cache[run_id]

    def _type_array(self, run_id, run):
        """The two extra node channels, or None when they were not asked for.

        "type" gives is_leader and is_preplanned. "noise" gives two random
        channels of the same width, seeded from the run id so it is
        reproducible. The noise arm is the capacity control: if the typed arm
        beats it, the gain came from the information rather than from simply
        having two more inputs.
        """
        if self.agent_types is None:
            return None
        n = run["positions"].shape[1]
        if self.agent_types == "type":
            return np.stack([run["leader_mask"], run["preplanned_mask"]],
                            axis=1).astype(np.float32)
        # a stable seed from the run id, because Python's hash() is salted per
        # process and would make the control irreproducible
        seed = int("".join(c for c in run_id if c.isdigit()) or 0)
        rng = np.random.default_rng(seed)
        return rng.random((n, 2), dtype=np.float32)

    def _special_nodes(self, run):
        """Which agents are leaders or preplanned, always, whatever the model sees.

        This is deliberately independent of `agent_types`. Which pairs are
        special is a property of the data, not of the model's inputs, and the
        untyped arm has to be scored on the same subset as the typed one for
        the comparison to mean anything. Choosing an evaluation subset leaks
        nothing into a model that was never given the flags.
        """
        return run["leader_mask"] | run["preplanned_mask"]

    def __len__(self):
        return len(self._index)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i):
        run_id, t = self._index[i]
        run = self._load(run_id)
        pos_t = run["positions"][t]
        vel_t = run["velocities"][t]

        x = feat_mod.node_features(run["positions"], run["velocities"],
                                   t, self.k_window,
                                   include_position=self.include_position,
                                   agent_types=self._type_array(run_id, run))
        pairs = cand_mod.candidate_pairs(pos_t, self.k_neighbors)
        y = self._label_matrix(pairs, pos_t, run["radii"])

        delta = pos_t[pairs[:, 0]] - pos_t[pairs[:, 1]]
        dvel = vel_t[pairs[:, 0]] - vel_t[pairs[:, 1]]
        edge_attr = np.stack([
            np.linalg.norm(delta, axis=-1),
            np.linalg.norm(dvel, axis=-1),
        ], axis=1)

        special_nodes = self._special_nodes(run)
        special = special_nodes[pairs[:, 0]] | special_nodes[pairs[:, 1]]

        return Sample(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(cand_mod.to_directed(pairs)),
            pairs=torch.from_numpy(pairs),
            edge_attr=torch.from_numpy(edge_attr.astype(np.float32)),
            y=torch.from_numpy(y),
            run_id=run_id,
            special=torch.from_numpy(special),
        )

    @staticmethod
    def _label_matrix(pairs, pos_t, radii):
        y = np.zeros((pairs.shape[0], len(label_mod.RELATIONS)),
                     dtype=np.float32)
        lookup = {tuple(p): row for row, p in enumerate(pairs)}
        for col, relation in enumerate(label_mod.RELATIONS):
            for pair in label_mod.edge_pairs(pos_t, radii[relation]):
                row = lookup.get(tuple(pair))
                # a true edge outside the candidate set is unreachable, which
                # is exactly what recall_report measures
                if row is not None:
                    y[row, col] = 1.0
        return y

    def recall_report(self):
        totals = {r: [0, 0] for r in label_mod.RELATIONS}
        for run_id, t in self._index:
            run = self._load(run_id)
            pos_t = run["positions"][t]
            cand = cand_mod.candidate_pairs(pos_t, self.k_neighbors)
            cand_set = {tuple(p) for p in cand}
            for relation in label_mod.RELATIONS:
                truth = label_mod.edge_pairs(pos_t, run["radii"][relation])
                totals[relation][1] += truth.shape[0]
                totals[relation][0] += sum(1 for p in truth
                                           if tuple(p) in cand_set)
        return {r: (hit / total if total else 0.0)
                for r, (hit, total) in totals.items()}
