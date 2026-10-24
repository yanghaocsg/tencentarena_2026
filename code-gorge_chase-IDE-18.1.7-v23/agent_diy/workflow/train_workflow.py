#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""v23: corner-cutting with full 3x3 terrain dump on every diagonal test.

Same local-corner method as v22, but each diagonal test prints the exact
3x3 neighborhood (rows 9-11, cols 9-11 of the view) plus the two edge
values and the target, so a BLOCKED result can be explained by the actual
terrain rather than inferred.
"""
import time

import numpy as np

from tools.train_env_conf_validate import read_usr_conf

DIRS = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
DIAG = [1, 3, 5, 7]
CARD = [0, 2, 4, 6]
NEED = 2
MAX_RESETS = 25
MAX_INNER = 80


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


def edges_for(view, d):
    """Return (target, edge1 (x+dx,z), edge2 (x,z+dz)) as ints."""
    dx, dz = DIRS[d]
    tgt = int(view[10 + dx, 10 + dz])
    e1 = int(view[10 + dx, 10])
    e2 = int(view[10, 10 + dz])
    return tgt, e1, e2


def is_corner(view, d):
    tgt, e1, e2 = edges_for(view, d)
    if tgt == 1 and ((e1 == 0 and e2 == 1) or (e1 == 1 and e2 == 0)):
        return tgt, e1, e2
    return None


def dump_3x3(view):
    for r in range(9, 12):
        out(f"    r{r} cols9-11 = {[int(v) for v in view[r, 9:12]]}")


def workflow(envs, agents, logger=None, monitor=None, *a, **k):
    out("PROBE v23 START")
    uc = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    env = envs[0]
    counts = {1: [0, 0], 3: [0, 0], 5: [0, 0], 7: [0, 0]}
    blocked_positions = set()
    tick = 0

    def done():
        return all(counts[d][0] + counts[d][1] >= NEED for d in DIAG)

    for r in range(MAX_RESETS):
        if done():
            break
        env_obs = env.reset(usr_conf=uc)
        cur = get_pos(env_obs["observation"]["frame_state"]["heroes"])
        for inner in range(MAX_INNER):
            if done():
                break
            obs = env_obs["observation"]
            view = parse_view(obs.get("map_info"))
            cur = get_pos(obs["frame_state"]["heroes"])

            action = None
            corner_info = None
            for d in DIAG:
                if counts[d][0] + counts[d][1] >= NEED:
                    continue
                ci = is_corner(view, d)
                if ci is not None and cur not in blocked_positions:
                    action = d
                    corner_info = ci
                    break
            if action is None:
                order = [CARD[(tick + i) % 4] for i in range(4)] + DIAG
                for d in order:
                    dx, dz = DIRS[d]
                    tr, tc = 10 + dx, 10 + dz
                    if 0 <= tr < 21 and 0 <= tc < 21 and view[tr, tc] == 1:
                        action = d
                        break
            if action is None:
                break

            before = cur
            if corner_info is not None:
                tgt, e1, e2 = corner_info
                out(f"DIAG act={action} at {before} targetVal={tgt} "
                    f"edge1(x+dx,z)={e1} edge2(x,z+dz)={e2}")
                dump_3x3(view)
            env_reward, env_obs = env.step(action)
            if "observation" not in env_obs:
                out("no observation, reset")
                break
            after = get_pos(env_obs["observation"]["frame_state"]["heroes"])
            moved = (after[0] - before[0], after[1] - before[1])
            if corner_info is not None:
                good = moved == DIRS[action]
                counts[action][0 if good else 1] += 1
                out(f"    -> moved={moved} expect={DIRS[action]} "
                    f"{'OK' if good else 'BLOCKED'} counts={counts[action]}")
                if not good:
                    blocked_positions.add(before)
            cur = after
            tick += 1
            if env_obs.get("terminated") or env_obs.get("truncated"):
                out(f"episode ended at {cur}, reset")
                break

    out("SUMMARY per diagonal [OK, BLOCKED]:")
    for d in DIAG:
        out(f"  act={d}: {counts[d]}")
    total_ok = sum(counts[d][0] for d in DIAG)
    total_bl = sum(counts[d][1] for d in DIAG)
    out(f"TOTAL OK={total_ok} BLOCKED={total_bl}")
    out("PROBE v23 END")
    time.sleep(3)
