#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""v19: direction + diagonal-corner-cutting calibration.

For each of the 8 directions, issue the action, then issue the opposite
to return, and print the actual (dx, dz) achieved plus the target cell's
gmap value. Also dump the full 21x21 raw view. This settles:
  - the exact direction of each action index;
  - whether a diagonal move needs BOTH adjacent edges passable (no corner
    cutting) or just one (documented "at least one").

The probe does not avoid the monster; it only needs the first ~16 steps,
well before any danger.
"""
import time

import numpy as np

from tools.train_env_conf_validate import read_usr_conf

GRID = 128
PASSABLE, BLOCKED, UNKNOWN = 1, 0, -1
DIRS = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
OPPOSITE = [4, 5, 6, 7, 0, 1, 2, 3]

# (action, label) pairs: go in a direction, then come back.
SEQUENCE = [
    (0, "right"), (4, "back-from-right"),
    (2, "up"), (6, "back-from-up"),
    (1, "UR"), (5, "back-from-UR"),
    (3, "UL"), (7, "back-from-UL"),
    (6, "down"), (2, "back-from-down"),
    (4, "left"), (0, "back-from-left"),
    (7, "DR"), (3, "back-from-DR"),
    (5, "DL"), (1, "back-from-DL"),
]


def out(msg):
    print(msg, flush=True)


def get_pos(d):
    if not d:
        return None
    p = d.get("pos")
    if not p:
        return None
    x, z = p.get("x"), p.get("z")
    if x is None or z is None or x < 0 or z < 0:
        return None
    return int(x), int(z)


def parse_view(mi):
    if not mi:
        return None
    if isinstance(mi[0], dict):
        return np.array([x["values"] for x in mi], dtype=np.int32)
    return np.array(mi, dtype=np.int32)


def dump_view(view):
    out("FULL VIEW (row=x, col=z; center=hero):")
    for ri in range(view.shape[0]):
        out(f"r{ri:02d} " + "".join(str(int(v)) for v in view[ri]))


def workflow(envs, agents, logger=None, monitor=None, *a, **k):
    out("PROBE v19 START")
    uc = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    env = envs[0]
    env_obs = env.reset(usr_conf=uc)

    obs = env_obs["observation"]
    fs = obs["frame_state"]
    hero0 = get_pos(fs["heroes"])
    mon0 = [get_pos(m) for m in fs.get("monsters", [])]
    mon0 = [m for m in mon0 if m]
    view = parse_view(obs.get("map_info"))
    out(f"hero={hero0} mon={mon0}")
    if view is not None:
        dump_view(view)

    cur = hero0
    for i, (act, label) in enumerate(SEQUENCE):
        dx, dz = DIRS[act]
        tx, tz = cur[0] + dx, cur[1] + dz
        tgt_in_map = "?"
        if 0 <= tx < GRID and 0 <= tz < GRID and view is not None:
            vr = 10 + (tx - hero0[0])
            vc = 10 + (tz - hero0[1])
            if 0 <= vr < 21 and 0 <= vc < 21:
                tgt_in_map = int(view[vr, vc])
        env_reward, env_obs = env.step(act)
        obs = env_obs["observation"]
        fs = obs["frame_state"]
        new = get_pos(fs["heroes"])
        moved = (new[0] - cur[0], new[1] - cur[1])
        out(f"#{i:02d} {label:16s} act={act} {cur} -> {new} "
            f"moved={moved} expect=({dx},{dz}) tgtRaw={tgt_in_map} "
            f"{'OK' if moved == (dx,dz) else 'BLOCKED'}")
        cur = new

    out("PROBE v19 END")
    time.sleep(3)
