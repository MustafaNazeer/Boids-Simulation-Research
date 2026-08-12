import json
import sys

from ml import baselines, paths, splits, train
from ml.data import features
from ml.data.npz_dataset import BoidsEdgeDataset
from ml.models.edge_gnn import EdgeClassifier


def run_comparison(run_dir, epochs=10, stride=10, seed=0, k_window=10,
                   k_neighbors=None):
    rows = splits.read_manifest(run_dir)
    train_ids, test_ids = splits.random_split(rows, 0.7, seed=seed)

    def build(ids, include_position=True):
        return BoidsEdgeDataset(run_dir, k_window=k_window,
                                k_neighbors=k_neighbors, stride=stride,
                                runs=ids, include_position=include_position)

    train_ds, test_ds = build(train_ids), build(test_ids)

    full = EdgeClassifier(use_distance=True)
    train.train(full, train_ds, epochs=epochs, seed=seed)

    # edge distance removed, but the node features still carry centroid
    # relative position, so the head can rebuild the distance from the two
    # embeddings. kept only to show that this arm proves nothing
    leaky = EdgeClassifier(use_distance=False)
    train.train(leaky, train_ds, epochs=epochs, seed=seed)

    # the genuine ablation: no edge distance and no positional node channels,
    # so the model sees velocity, speed, and relative speed only
    nopos_train = build(train_ids, include_position=False)
    nopos_test = build(test_ids, include_position=False)
    honest = EdgeClassifier(use_distance=False,
                            in_dim=features.feature_dim(False))
    train.train(honest, nopos_train, epochs=epochs, seed=seed)

    return {
        "model": train.evaluate(full, test_ds),
        "ablation_leaky": train.evaluate(leaky, test_ds),
        "ablation_no_position": train.evaluate(honest, nopos_test),
        "distance": baselines.distance_reference(test_ds),
        "oracle": baselines.oracle_threshold_reference(test_ds),
        "recall": test_ds.recall_report(),
    }


def main():
    out = run_comparison(paths.fixture_dir(), epochs=10, stride=10)
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
