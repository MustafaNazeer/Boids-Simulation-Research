import json
import sys
import time

from ml import arms, baselines, paths, splits, train
from ml.data import features
from ml.data.npz_dataset import BoidsEdgeDataset
from ml.models.edge_gnn import EdgeClassifier


def held_out_bounds(modes):
    """A split that holds out whole world types rather than whole runs.

    Returns a callable with the same shape as the default random split, so it
    can be handed straight to run_comparison. The bounds column is categorical,
    which is why splits.split_by_column cannot do this.
    """
    def split(rows, seed):
        return splits.split_by_category(rows, "bounds", modes)
    return split


def split_within(exclude_modes, fraction=0.8):
    """Train and test both drawn only from the worlds NOT excluded.

    The excluded worlds are left untouched so they can be handed to
    run_comparison as extra_test_ids. That is what lets one trained model be
    scored both in distribution and out of distribution, with the test data as
    the only thing that differs between the two numbers.
    """
    exclude = set(exclude_modes)

    def split(rows, seed):
        kept = [r for r in rows if r["bounds"] not in exclude]
        return splits.random_split(kept, fraction, seed)
    return split


def ids_with_bounds(rows, modes):
    modes = set(modes)
    return sorted(splits._run_id(r) for r in rows if r["bounds"] in modes)


def run_comparison(run_dir, epochs=10, stride=10, seed=0, k_window=10,
                   k_neighbors=None, split=None, extra_test_ids=None):
    rows = splits.read_manifest(run_dir)
    if split is None:
        train_ids, test_ids = splits.random_split(rows, 0.7, seed=seed)
    else:
        train_ids, test_ids = split(rows, seed)

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

    def score(ids):
        # the same three trained models, scored against a different test set.
        # nothing is retrained here, which is the point: the only thing that
        # differs between this and the in distribution report is the data
        ds = build(ids)
        ds_nopos = build(ids, include_position=False)
        return {
            "model": train.evaluate(full, ds),
            "ablation_leaky": train.evaluate(leaky, ds),
            "ablation_no_position": train.evaluate(honest, ds_nopos),
            "distance": baselines.distance_reference(ds),
            "oracle": baselines.oracle_threshold_reference(ds),
            "recall": ds.recall_report(),
            "n_runs": len(ds.runs),
        }

    out = {
        "model": train.evaluate(full, test_ds),
        "ablation_leaky": train.evaluate(leaky, test_ds),
        "ablation_no_position": train.evaluate(honest, nopos_test),
        "distance": baselines.distance_reference(test_ds),
        "oracle": baselines.oracle_threshold_reference(test_ds),
        "recall": test_ds.recall_report(),
    }
    if extra_test_ids is not None:
        out["transfer"] = score(extra_test_ids)
        out["n_train_runs"] = len(train_ds.runs)
        out["n_test_runs"] = len(test_ds.runs)
    return out


def run_agent_type_comparison(run_dir, epochs=10, stride=10, seed=0,
                              k_window=10, k_neighbors=None):
    """Does telling the model which agents are leaders or preplanned help?

    Every arm here drops position, because that is the only variant where the
    model must infer relationships from motion rather than read them off
    geometry, and therefore the only place agent type could plausibly matter.

    Three arms, identical but for the two extra node channels:
      plain  no extra channels, the current 0.93 AUC arm
      typed  is_leader and is_preplanned
      noise  two reproducible random channels

    The noise arm is the capacity control. Without it, an improvement from
    `typed` could not be distinguished from simply having two more inputs.

    Each arm is scored twice: over all pairs, and over only those pairs
    touching a special agent. Only about 4.5 percent of agents are special, so
    the overall metric is dominated by ordinary pairs and cannot show an
    effect confined to the rest. The untyped arms are scored on the same
    subset; the flags choose which pairs to score and are never fed to those
    models.
    """
    rows = splits.read_manifest(run_dir)
    train_ids, test_ids = splits.random_split(rows, 0.7, seed=seed)

    def build(ids, agent_types):
        return BoidsEdgeDataset(run_dir, k_window=k_window,
                                k_neighbors=k_neighbors, stride=stride,
                                runs=ids, include_position=False,
                                agent_types=agent_types)

    out = {}
    for name, kind in (("plain", None), ("typed", "type"), ("noise", "noise")):
        n_extra = 0 if kind is None else 2
        tr, te = build(train_ids, kind), build(test_ids, kind)
        model = EdgeClassifier(
            use_distance=False,
            in_dim=features.feature_dim(False, n_extra))
        train.train(model, tr, epochs=epochs, seed=seed)
        out[name] = {
            "overall": train.evaluate(model, te),
            "special": train.evaluate(model, te, only_special=True),
        }

    # context for reading the overall numbers: what fraction of scored pairs
    # could the type channels possibly have informed
    ref = build(test_ids, None)
    total = touching = 0
    for sample in ref:
        n = int(sample.special.shape[0])
        total += n
        touching += int(sample.special.sum())
    out["special_pair_share"] = (touching / total) if total else 0.0
    return out


def main():
    out = run_comparison(paths.fixture_dir(), epochs=10, stride=10)
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()


def architecture_arm_names():
    return list(arms.ARMS)


def build_architecture_model(arm, include_position=False):
    """One model for one arm, always position free.

    Kept separate from run_architecture_comparison so a test can assert the
    position free property without training anything.
    """
    return EdgeClassifier.from_arm(
        arm, in_dim=features.feature_dim(include_position),
        use_distance=include_position)


def run_architecture_comparison(run_dir, arm_names=None, epochs=10, stride=10,
                                seed=0):
    """Train one model per arm on identical data and score them together.

    Every arm is position free. The full feature task sits at about 0.999 with
    distance alone at the same number, so architectures compared there would
    all land inside each other's noise. The position free configuration has
    roughly 45 points of headroom, which is the only place in this project
    where one architecture can be told apart from another.

    Datasets are cached per window length rather than rebuilt per arm, because
    only the k_window axis changes the data at all and rebuilding it thirteen
    times would dominate the run.
    """
    names = architecture_arm_names() if arm_names is None else list(arm_names)
    unknown = [n for n in names if n not in arms.ARMS]
    if unknown:
        # raised before any training, so a typo costs a second rather than an
        # hour
        raise KeyError("unknown arm name(s): %s" % ", ".join(unknown))

    rows = splits.read_manifest(run_dir)
    train_ids, test_ids = splits.random_split(rows, 0.7, seed=seed)

    cache = {}

    def datasets(k_window):
        if k_window not in cache:
            def build(ids):
                return BoidsEdgeDataset(run_dir, k_window=k_window,
                                        k_neighbors=None, stride=stride,
                                        runs=ids, include_position=False)
            cache[k_window] = (build(train_ids), build(test_ids))
        return cache[k_window]

    out = {}
    for name in names:
        arm = arms.ARMS[name]
        tr, te = datasets(arm.k_window)
        model = build_architecture_model(arm)
        started = time.time()
        train.train(model, tr, epochs=epochs, seed=seed)
        elapsed = time.time() - started
        out[name] = {
            "scores": train.evaluate(model, te),
            "params": sum(p.numel() for p in model.parameters()),
            "seconds": elapsed,
        }
    return out
