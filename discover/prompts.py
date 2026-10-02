"""The blind prompt. It never says what the agents are.

The model may have read papers about the real data in training. Naming the
system would let it recall rather than search, so the prompt describes only
abstract agents in 3D, and a test asserts the forbidden words never appear.
"""

import ast
import re
from collections import Counter

import numpy as np

FORBIDDEN_WORDS = ("midge", "insect", "swarm", "flock", "boid", "separation",
                   "alignment", "cohesion", "chironomus", "mosquito", "bird",
                   "fish", "kelley", "ouellette", "sinhuber", "vicsek")

SYSTEM = """You are helping discover the interaction rule that governs a group \
of agents moving in three dimensions. Each agent's acceleration is assumed to \
be a weighted sum of a few feature terms that you design. You write the terms; \
the weights are fitted for you by least squares, and up to three tunable \
scalar parameters you declare are searched for you within the ranges you give.

Write exactly one Python program in a single ```python fenced block. Rules:
1. Only `import numpy as np` and `import math` are allowed. No files, no I/O.
2. Define PARAMS as a literal dict mapping each tunable parameter name to a \
(low, high) tuple, at most three entries. Use PARAMS = {} if you need none.
3. Define features(pos, vel, center, params). pos and vel are (N, 3) arrays \
of the agents present in one frame, center is the (3,) mean of pos, and \
params maps your parameter names to floats. Return an (N, 3, K) array with \
1 <= K <= 6: one 3D vector per agent per term. Every value must be finite.
4. Programs are scored by how well the fitted rule predicts held out data, \
minus a small penalty for each term and each parameter, so prefer simple \
rules that explain the data.
5. Physical units: positions are in the data's length unit and time in \
seconds. Choose parameter ranges accordingly.
6. Name every distance parameter r or r_<name>, and no other parameter that \
way, because distance parameters have a data driven minimum and a lower \
bound below it is raised to that minimum."""


def _names_forbidden(text):
    lower = text.lower()
    return any(word in lower for word in FORBIDDEN_WORDS)


def _drop_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return tree


def clean_source(source):
    """The program as the model may see it, or None if it must not be shown.

    Comments and docstrings go (ast.unparse drops comments). A program that
    still names a forbidden word anywhere, in an identifier or a string, is
    not shown at all.
    """
    try:
        cleaned = ast.unparse(_drop_docstrings(ast.parse(source)))
    except (SyntaxError, ValueError):
        return None
    lower = cleaned.lower()
    if any(word in lower for word in FORBIDDEN_WORDS):
        return None
    return cleaned


def _num(x):
    return "%.4g" % x


def scale_facts(fit_windows, radius_floor):
    """Neutral facts about the data's scale, from the fit windows only.

    Without them the model cannot know whether a distance of 1 is tiny or
    huge, and on real data it guessed ranges far below the radius floor.
    Units are never named: lengths are in the data's length unit.
    """
    dts, nearest, pairs, speeds, counts = [], [], [], [], []
    for w in fit_windows:
        dts.append(w.dt)
        for t in range(w.positions.shape[0]):
            present = ~np.isnan(w.positions[t, :, 0])
            p = w.positions[t][present]
            counts.append(len(p))
            v = w.velocities[t][present]
            s = np.linalg.norm(v, axis=1)
            speeds.append(s[np.isfinite(s)])
            if len(p) < 2:
                continue
            dist = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
            pairs.append(dist[np.triu_indices(len(p), 1)])
            np.fill_diagonal(dist, np.inf)
            nearest.append(dist.min(axis=1))
    return ("Scale of the data, measured on the fit windows. Lengths are in "
            "the data's length unit and time is in seconds. Frames are %s "
            "seconds apart. A typical frame has %s agents present. The median "
            "distance from an agent to its nearest neighbor is %s length "
            "units, and 90%% of distances between pairs of agents are below "
            "%s length units. The median agent speed is %s length units per "
            "second. The minimum for distance parameters is %s length units, "
            "so a distance range must extend above it."
            % (_num(float(np.median(dts))), _num(float(np.median(counts))),
               _num(float(np.median(np.concatenate(nearest)))),
               _num(float(np.percentile(np.concatenate(pairs), 90))),
               _num(float(np.median(np.concatenate(speeds)))),
               _num(float(radius_floor))))


def _failure_text(failures):
    """Up to three of the most common reasons, never one naming a forbidden
    word, each cut to 200 characters."""
    counts = Counter()
    for reason in failures or ():
        reason = " ".join(str(reason).split())
        lower = reason.lower()
        if not reason or any(word in lower for word in FORBIDDEN_WORDS):
            continue
        counts[reason[:200]] += 1
    if not counts:
        return ""
    lines = ["- %s (%d program%s)" % (r, n, "" if n == 1 else "s")
             for r, n in counts.most_common(3)]
    return ("Recent programs failed for these reasons:\n" + "\n".join(lines)
            + "\n\n")


def build(best, failures=None, scale_text=""):
    head = scale_text.strip() + "\n\n" if scale_text.strip() else ""
    head += _failure_text(failures)
    shown_best = []
    for source, score, val_r2 in best:
        cleaned = clean_source(source)
        # clean_source already refuses a forbidden word; this second scan of
        # exactly the text that would be shown keeps the guard in place even
        # if the cleaning step changes
        if cleaned is not None and not _names_forbidden(cleaned):
            shown_best.append((cleaned, score, val_r2))
    if not shown_best:
        shown = "No programs have been evaluated yet. Propose a first rule."
    else:
        parts = []
        for i, (source, score, val_r2) in enumerate(shown_best[:3], 1):
            parts.append("Program %d: score %.4f, held out R^2 %.4f\n"
                         "```python\n%s\n```" % (i, score, val_r2,
                                                  source.strip()))
        shown = ("The best programs so far, best first:\n\n"
                 + "\n\n".join(parts)
                 + "\n\nPropose one new program that you expect to score "
                   "higher. You may refine one of these or try a different "
                   "idea.")
    return head + shown


def parse_program(text):
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    return blocks[-1] if blocks else None
