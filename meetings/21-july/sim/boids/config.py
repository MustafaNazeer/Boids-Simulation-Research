import os
import copy
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.normpath(os.path.join(HERE, "..", "config.yaml"))

BOUNDS_MODES = ("reflect", "wrap")

RULE_KEYS = {
    "separation": ("radius", "weight"),
    "alignment": ("radius", "weight"),
    "cohesion": ("radius", "weight"),
    "obstacle_avoidance": ("range", "weight"),
    "predator_flee": ("radius", "weight"),
    "leader_follow": ("radius", "weight"),
    "leader_wander": ("strength", "weight"),
}

WEIGHT_NAMES = {
    "separation": "separation",
    "alignment": "alignment",
    "cohesion": "cohesion",
    "obstacle_avoidance": "obstacle",
    "predator_flee": "flee",
    "leader_follow": "follow",
    "leader_wander": "wander",
}

DEFAULTS = {
    "world": {"size": 60.0, "bounds": "reflect", "n_boids": 120, "seed": 11},
    "motion": {"max_speed": 5.0, "min_speed": 1.5, "dt": 0.2, "frames": 300},
    "rules": {
        "separation": {"enabled": True, "radius": 7.0, "weight": 8.0},
        "alignment": {"enabled": True, "radius": 16.0, "weight": 1.6},
        "cohesion": {"enabled": True, "radius": 16.0, "weight": 0.5},
        "obstacle_avoidance": {"enabled": True, "range": 4.0, "weight": 2.5},
        "predator_flee": {"enabled": True, "radius": 15.0, "weight": 3.5},
        "leader_follow": {"enabled": True, "radius": 20.0, "weight": 1.0},
        "leader_wander": {"enabled": True, "strength": 0.15, "weight": 2.0},
    },
    "leaders": {"count": 3},
    "preplanned": {"count": 0, "speed": 5.0},
    "scenario": {"predator": True, "obstacles": True},
}


def _merge(base, override, path=""):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        where = "%s.%s" % (path, key) if path else key
        if key not in out:
            raise ValueError("unknown config key: %s" % where)
        if isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _merge(out[key], value, where)
        else:
            out[key] = value
    return out


def load(path=None):
    path = DEFAULT_PATH if path is None else path
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, raw)
    validate(cfg)
    return cfg


def validate(cfg):
    bounds = cfg["world"]["bounds"]
    if bounds not in BOUNDS_MODES:
        raise ValueError("world.bounds must be one of %s, got %r"
                         % (list(BOUNDS_MODES), bounds))
    if cfg["world"]["n_boids"] < 1:
        raise ValueError("world.n_boids must be at least 1")
    if cfg["motion"]["min_speed"] > cfg["motion"]["max_speed"]:
        raise ValueError("motion.min_speed cannot exceed motion.max_speed")
    total = cfg["leaders"]["count"] + cfg["preplanned"]["count"]
    if total > cfg["world"]["n_boids"]:
        raise ValueError(
            "leaders.count plus preplanned.count (%d) exceeds world.n_boids (%d)"
            % (total, cfg["world"]["n_boids"]))
    for name, rule in cfg["rules"].items():
        for field in RULE_KEYS[name]:
            if rule[field] < 0:
                raise ValueError("rules.%s.%s must not be negative"
                                 % (name, field))
    return cfg


def to_params(cfg):
    rules = cfg["rules"]
    params = {
        "separation_radius": rules["separation"]["radius"],
        "alignment_radius": rules["alignment"]["radius"],
        "cohesion_radius": rules["cohesion"]["radius"],
        "max_speed": cfg["motion"]["max_speed"],
        "min_speed": cfg["motion"]["min_speed"],
        "preplanned_speed": cfg["preplanned"]["speed"],
        "obstacle_avoid_range": rules["obstacle_avoidance"]["range"],
        "fear_radius": rules["predator_flee"]["radius"],
        "leader_follow_radius": rules["leader_follow"]["radius"],
        "leader_wander_strength": rules["leader_wander"]["strength"],
        "weights": {WEIGHT_NAMES[name]: rules[name]["weight"]
                    for name in RULE_KEYS},
        "enabled": {name: bool(rules[name]["enabled"]) for name in RULE_KEYS},
    }
    return params
