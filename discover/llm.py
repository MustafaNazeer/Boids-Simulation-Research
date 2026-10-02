"""The only module that talks to the Claude API.

Requests go through the Message Batches API at half price. The server side
refusal fallback is not used because the Batches API rejects it; a refusal
is reported as a status and the caller logs it as a failed candidate.
"""

import time
from dataclasses import dataclass

from discover.ledger import Usage

MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 4000
EFFORT = "medium"
# Haiku 4.5 rejects the effort parameter, so it is sent only to these models.
# No model is sent a thinking parameter: Sonnet 5.5 and Opus 5.5 run their
# adaptive default, and Haiku 4.5 runs without extended thinking.
EFFORT_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5"}
NOT_FOUND_ATTEMPTS = 10
# The server expires a batch 24 hours after creation, so a batch still
# unfinished two hours past that will never finish. Polling stops there.
DEADLINE_SECONDS = 26 * 3600


@dataclass
class Result:
    custom_id: str
    status: str
    text: str
    usage: Usage
    stop_reason: str = ""


def _usage(u):
    return Usage(input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                 cache_creation_input_tokens=u.cache_creation_input_tokens or 0,
                 cache_read_input_tokens=u.cache_read_input_tokens or 0)


class BatchClient:
    def __init__(self, client=None, poll_seconds=30, model=MODEL,
                 deadline_seconds=DEADLINE_SECONDS, clock=time.monotonic):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.poll = poll_seconds
        self.model = model
        self.deadline = deadline_seconds
        self.clock = clock

    def _request(self, custom_id, system, prompt):
        params = {"model": self.model, "max_tokens": MAX_TOKENS,
                  "system": system,
                  "messages": [{"role": "user", "content": prompt}]}
        if self.model in EFFORT_MODELS:
            params["output_config"] = {"effort": EFFORT}
        return {"custom_id": custom_id, "params": params}

    def _retry(self, call, batch_id):
        """A batch can 404 for a moment right after creation. Retry only that."""
        import anthropic
        for attempt in range(NOT_FOUND_ATTEMPTS):
            try:
                return call(batch_id)
            except anthropic.NotFoundError:
                if attempt == NOT_FOUND_ATTEMPTS - 1:
                    raise
                time.sleep(self.poll)

    def run(self, items, on_created=None):
        batches = self.client.messages.batches
        batch = batches.create(requests=[self._request(*i) for i in items])
        if on_created is not None:
            on_created(batch.id)
        start = self.clock()
        while self._retry(batches.retrieve, batch.id).processing_status != "ended":
            if self.clock() - start >= self.deadline:
                raise TimeoutError("batch %s still unfinished after %g seconds"
                                   % (batch.id, self.deadline))
            time.sleep(self.poll)
        out = []
        for r in self._retry(lambda b: list(batches.results(b)), batch.id):
            if r.result.type == "succeeded":
                msg = r.result.message
                text = "".join(b.text for b in msg.content if b.type == "text")
                status = "refusal" if msg.stop_reason == "refusal" else "succeeded"
                out.append(Result(r.custom_id, status, text, _usage(msg.usage),
                                  msg.stop_reason or ""))
            else:
                out.append(Result(r.custom_id, r.result.type, "", Usage(0, 0),
                                  r.result.type))
        return out, batch.id


class FakeBatchClient:
    """Stands in for the API in every test. Spends nothing."""

    def __init__(self, reply, tokens=(1500, 1500), model=MODEL):
        self.reply = reply
        self.model = model
        self.tokens = tokens
        self.calls = 0

    def run(self, items, on_created=None):
        self.calls += 1
        batch_id = "fake-%d" % self.calls
        if on_created is not None:
            on_created(batch_id)
        out = [Result(cid, "succeeded", self.reply(prompt),
                      Usage(self.tokens[0], self.tokens[1]))
               for cid, _system, prompt in items]
        return out, batch_id
