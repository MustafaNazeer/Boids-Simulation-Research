"""The thirteen architecture arms, as data rather than as branches.

Every arm changes exactly one thing against the reference, which is what makes
each measured number attributable to a single change. `test_arms.py` enforces
that property rather than trusting it.

Three arms are controls rather than candidates. `enc_meanpool` and `enc_mlp`
together answer the question `meetings/august/11/gru-notes.md` names as the
hardest one available, what a GRU buys over a 10 frame window: meanpool removes
recurrence and frame ordering together, mlp removes only recurrence, so the two
read jointly separate the causes. `rel_none` asks whether message passing
contributes anything at all on this task.
"""

import dataclasses

# which model field each axis is allowed to vary. the test uses this to prove
# no arm changes two things at once
AXIS_FIELD = {
    "encoder": "encoder",
    "relational": "relational",
    "hidden": "hidden",
    "k_window": "k_window",
}


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    axis: str
    encoder: str
    relational: str
    hidden: int
    k_window: int


def _arm(name, axis, **overrides):
    base = dict(encoder="gru", relational="mp1", hidden=64, k_window=10)
    base.update(overrides)
    return Arm(name=name, axis=axis, **base)


REFERENCE = _arm("reference", "reference")

_ALL = [
    REFERENCE,

    # does recurrence earn its place, and is the GRU the right recurrence
    _arm("enc_meanpool", "encoder", encoder="meanpool"),
    _arm("enc_mlp", "encoder", encoder="mlp"),
    _arm("enc_lstm", "encoder", encoder="lstm"),
    _arm("enc_transformer", "encoder", encoder="transformer"),

    # does the graph half earn its place, and does attention beat mean
    # aggregation
    _arm("rel_none", "relational", relational="none"),
    _arm("rel_mp2", "relational", relational="mp2"),
    _arm("rel_gat", "relational", relational="gat"),

    # how much history the task actually needs
    _arm("k5", "k_window", k_window=5),
    _arm("k20", "k_window", k_window=20),
    _arm("k40", "k_window", k_window=40),

    # is capacity or architecture the binding constraint
    _arm("hidden32", "hidden", hidden=32),
    _arm("hidden128", "hidden", hidden=128),
]

ARMS = {arm.name: arm for arm in _ALL}


def axis_members(axis):
    """Every arm on one axis, excluding the reference."""
    return [arm for arm in _ALL if arm.axis == axis]
