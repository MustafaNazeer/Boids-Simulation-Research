"""Every paid request passes through here before it is sent.

The file is append only. A reservation counts at its worst case until its
batch is settled with the usage the API reported, so a crash between the two
can only overstate spend, never understate it.

Input tokens are bounded at one token per byte of UTF-8. The input_chars
argument is that byte count, and no tokenizer emits more tokens than the
bytes it reads. Prompts here are ASCII code and prose, which run nearer four
bytes per token, so the bound is several times the likely cost.

A torn last line from a crash (incomplete json.dumps write) makes the ledger
fail closed because json.loads raises on the invalid line, which is intended
and safe.
"""

import json
from dataclasses import dataclass

# USD per million tokens, input and output, standard (unbatched) rates, from
# Anthropic's published API pricing as of 2026-09-25.
PRICES = {"claude-haiku-4-5": (1.00, 5.00),
          "claude-sonnet-5-5": (2.00, 10.00),
          "claude-opus-5-5": (4.00, 20.00)}
BATCH_DISCOUNT = 0.5
CACHE_WRITE = 1.25
CACHE_READ = 0.1


class OverBudget(Exception):
    pass


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class Ledger:
    def __init__(self, path, cap_usd):
        self.path = path
        self.cap = float(cap_usd)

    def _entries(self):
        try:
            with open(self.path) as f:
                return [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            return []

    def _append(self, entry):
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def committed(self, task=None):
        cost = {}
        for e in self._entries():
            if task is not None and e["task"] != task:
                continue
            cost[e["key"]] = e["usd"]
        return sum(cost.values())

    def reserve(self, key, task, task_cap, model, n_requests, input_chars,
                max_tokens):
        if any(e["key"] == key for e in self._entries()):
            raise ValueError("key already in ledger: %s" % key)
        pin, pout = PRICES[model]
        # input_chars is a UTF-8 byte count, bounded at one token per byte
        worst = (input_chars * CACHE_WRITE * pin
                 + n_requests * max_tokens * pout) / 1e6 * BATCH_DISCOUNT
        if self.committed() + worst > self.cap:
            raise OverBudget("batch %s could cost %.4f USD, past the %.2f USD "
                             "cap with %.4f committed"
                             % (key, worst, self.cap, self.committed()))
        if self.committed(task) + worst > task_cap:
            raise OverBudget("batch %s could cost %.4f USD, past the %.2f USD "
                             "cap for %s" % (key, worst, task_cap, task))
        self._append({"key": key, "task": task, "kind": "reserve",
                      "model": model, "usd": worst})
        return worst

    def settle(self, key, model, usages):
        entries = self._entries()
        reserve_entry = next((e for e in entries if e["key"] == key and e["kind"] == "reserve"), None)
        if reserve_entry is None:
            raise ValueError("no reservation for key: %s" % key)
        if any(e["key"] == key and e["kind"] == "settle" for e in entries):
            raise ValueError("already settled: %s" % key)
        reserved = reserve_entry.get("model")
        if reserved is not None and reserved != model:
            raise ValueError("batch %s was reserved under %s, not %s"
                             % (key, reserved, model))
        pin, pout = PRICES[model]
        tokens_in = sum(u.input_tokens
                        + CACHE_WRITE * u.cache_creation_input_tokens
                        + CACHE_READ * u.cache_read_input_tokens
                        for u in usages)
        tokens_out = sum(u.output_tokens for u in usages)
        actual = (tokens_in * pin + tokens_out * pout) / 1e6 * BATCH_DISCOUNT
        task = reserve_entry["task"]
        self._append({"key": key, "task": task, "kind": "settle",
                      "model": model, "usd": actual})
        return actual
