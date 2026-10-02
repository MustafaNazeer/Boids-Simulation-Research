"""Run one LLM written candidate in a limited subprocess.

A static check refuses anything outside numpy and math before any process
starts. The child then runs with a clean environment and a wall clock
timeout, and child.py caps its own address space at 2 GiB and forbids file
writes before it touches the candidate. The limits are set in the child
rather than through preexec_fn, which is unsafe once search.py evaluates
candidates from several threads. It does not block the network at the
operating system level; the import allowlist is what keeps network and file
access out, which suits a model asked for numeric code rather than an
adversary.
"""

import ast
import json
import os
import subprocess
import sys
import tempfile

ALLOWED_IMPORTS = {"numpy", "math"}
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "input", "globals",
                   "locals", "vars", "getattr", "setattr", "delattr",
                   "breakpoint", "memoryview"}
FORBIDDEN_ATTRS = {"load", "save", "savez", "savez_compressed", "savetxt",
                   "loadtxt", "genfromtxt", "fromfile", "tofile", "memmap",
                   "ctypeslib", "lib", "f2py", "testing", "distutils",
                   "fromregex", "DataSource", "dump"}
SENTINEL = "@@DISCOVER_RESULT@@"
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Rejected(Exception):
    pass


def check_source(source):
    try:
        source.encode("utf-8")
    except UnicodeEncodeError:
        raise Rejected("source is not valid UTF-8 text")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise Rejected("syntax error: %s" % exc.msg)
    except (RecursionError, MemoryError, ValueError) as exc:
        raise Rejected("source cannot be parsed: %s" % type(exc).__name__)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise Rejected("import of %s is not allowed" % alias.name)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_IMPORTS:
                raise Rejected("import from %s is not allowed" % node.module)
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            raise Rejected("dunder name %s is not allowed" % node.id)
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise Rejected("dunder attribute %s is not allowed" % node.attr)
            if node.attr in FORBIDDEN_ATTRS:
                raise Rejected("attribute %s is not allowed" % node.attr)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in FORBIDDEN_CALLS):
            raise Rejected("call to %s is not allowed" % node.func.id)
    top = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "features" not in top:
        raise Rejected("no top level features function")
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "PARAMS"):
            try:
                value = ast.literal_eval(node.value)
            except Exception:
                raise Rejected("PARAMS must be a literal dict")
            if not isinstance(value, dict):
                raise Rejected("PARAMS must be a literal dict")


def evaluate(source, fit_path, val_path, timeout=10.0):
    try:
        check_source(source)
    except Rejected as exc:
        return {"ok": False, "reason": "rejected: %s" % exc}
    with tempfile.TemporaryDirectory() as tmp:
        cand = os.path.join(tmp, "candidate.py")
        with open(cand, "w", encoding="utf-8") as f:
            f.write(source)
        env = {"PYTHONPATH": PACKAGE_ROOT, "PYTHONDONTWRITEBYTECODE": "1",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "PATH": "/usr/bin:/bin"}
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "discover.child", cand,
                 os.path.abspath(fit_path), os.path.abspath(val_path)],
                cwd=tmp, env=env, capture_output=True, text=True,
                timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": "timeout"}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(SENTINEL)]
    if lines:
        try:
            out = json.loads(lines[-1][len(SENTINEL):])
        except ValueError:
            out = None
        if isinstance(out, dict) and "ok" in out:
            return out
        return {"ok": False, "reason": "crashed: unparseable output"}
    tail = proc.stderr.strip().splitlines()[-1:] or ["no output"]
    return {"ok": False, "reason": "crashed: %s" % tail[0]}
