"""Island search: the model proposes, the sandbox scores, the log records.

Each generation is one batch across all islands. Every island shows the
model its own best three programs, so the islands explore independently.
Nothing is retried: a refusal, an error or a reply without code becomes a
failed candidate with its reason in the log. The next generation of the same
island is shown the most common of those reasons, and every prompt carries
the data's scale (scale_text), so the model can size its parameters.
"""

import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from discover import ledger as ledger_mod
from discover import llm, prompts, sandbox


@dataclass
class Candidate:
    id: str
    island: int
    generation: int
    source: str
    ok: bool
    reason: str = ""
    score: float = float("-inf")
    val_r2: float = float("nan")
    weights: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    excluded_from_prompt: bool = False
    # the parameter ranges actually searched, after the radius floor
    ranges: dict = field(default_factory=dict)
    radius_floor: float | None = None


def _log(path, entry):
    entry["time"] = time.time()
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _loggable(c):
    """The candidate as a log entry, in strict JSON.

    A failed candidate keeps score -inf and val_r2 NaN in memory so it sorts
    last, but json.dumps would write those as -Infinity and NaN, which strict
    parsers refuse. The log records them as null instead.
    """
    entry = asdict(c)
    for name in ("score", "val_r2"):
        value = entry[name]
        if isinstance(value, float) and not math.isfinite(value):
            entry[name] = None
    return entry


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _top(cands, k=3):
    ok = sorted((c for c in cands if c.ok and not c.excluded_from_prompt),
                key=lambda c: -c.score)
    return [(c.source, c.score, c.val_r2) for c in ok[:k]]


def run(task, task_cap, fit_path, val_path, client, ledger, log_path,
        generations, islands=4, per_island=4, seed=0, timeout=10.0,
        scale_text=""):
    # Batch keys must be unique in the ledger, and one task is searched more
    # than once, so each run gets its own id.
    run_id = str(time.time_ns())
    _log(log_path, {"type": "start", "task": task, "model": client.model,
                    "run_id": run_id, "generations": generations,
                    "islands": islands, "per_island": per_island,
                    "seed": seed, "fit_path": os.path.abspath(fit_path),
                    "fit_sha256": _sha256(fit_path),
                    "val_path": os.path.abspath(val_path),
                    "val_sha256": _sha256(val_path),
                    "scale_text": scale_text})
    population = {i: [] for i in range(islands)}
    found = []
    for g in range(generations):
        items = []
        for i in range(islands):
            failures = [c.reason for c in population[i]
                        if c.generation == g - 1 and not c.ok]
            prompt = prompts.build(_top(population[i]), failures, scale_text)
            for k in range(per_island):
                items.append(("%s-g%d-i%d-k%d" % (task, g, i, k),
                              prompts.SYSTEM, prompt))
        key = "%s-%s-g%d-seed%d" % (task, run_id, g, seed)
        # UTF-8 bytes, which the ledger bounds at one token each
        chars = sum(len(s.encode("utf-8")) + len(p.encode("utf-8"))
                    for _, s, p in items)
        try:
            ledger.reserve(key, task, task_cap, client.model, len(items), chars,
                           llm.MAX_TOKENS)
        except ledger_mod.OverBudget as exc:
            _log(log_path, {"type": "stopped", "reason": str(exc)})
            break
        for cid, system, prompt in items:
            _log(log_path, {"type": "request", "id": cid, "prompt": prompt})
        # Log the batch id the moment it exists, so a crash while polling
        # still leaves a way to find the batch and its cost.
        results, batch_id = client.run(
            items, on_created=lambda bid: _log(log_path, {
                "type": "batch_created", "batch_id": bid, "key": key}))
        usd = ledger.settle(key, client.model, [r.usage for r in results])
        _log(log_path, {"type": "batch", "key": key, "batch_id": batch_id,
                        "usd": usd})

        def judge(r):
            island = int(r.custom_id.rsplit("-i", 1)[1].split("-")[0])
            source = ""
            try:
                if r.status != "succeeded":
                    return Candidate(r.custom_id, island, g, "", False,
                                     "result status %s" % r.status)
                source = prompts.parse_program(r.text)
                if source is None:
                    return Candidate(r.custom_id, island, g, "", False,
                                     "no python block in reply")
                out = sandbox.evaluate(source, fit_path, val_path, timeout)
                if not out["ok"]:
                    return Candidate(r.custom_id, island, g, source, False,
                                     out["reason"])
                return Candidate(r.custom_id, island, g, source, True, "",
                                 out["score"], out["val_r2"], out["weights"],
                                 out["params"], ranges=out["ranges"],
                                 radius_floor=out["radius_floor"])
            except Exception as exc:
                return Candidate(r.custom_id, island, g, source or "", False,
                                 "evaluator error: %s: %s"
                                 % (type(exc).__name__, exc))

        with ThreadPoolExecutor(max_workers=4) as pool:
            judged = list(pool.map(judge, results))
        for r, c in zip(results, judged):
            _log(log_path, {"type": "result", "id": r.custom_id,
                            "status": r.status,
                            "stop_reason": r.stop_reason, "text": r.text,
                            "usage": asdict(r.usage)})
            # A program that still names a forbidden word after comments and
            # docstrings are stripped is scored and logged but never fed back.
            if c.ok and prompts.clean_source(c.source) is None:
                c.excluded_from_prompt = True
            _log(log_path, dict(type="candidate", **_loggable(c)))
            population[c.island].append(c)
            found.append(c)
    return found
