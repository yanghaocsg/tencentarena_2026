#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""v24: corner-cutting + monster cross-check.

Same as v23 but every diagonal test also prints the visible monsters, and
on BLOCKED it explicitly reports whether a monster is at / adjacent to the
target cell. This tests whether v23's single BLOCKED was a monster
collision rather than a corner-cutting rule.
"""
import time

import numpy as np

from tools.train_env_conf_validate import read_usr_conf

DIRS = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
DIAG = [1, 3, 5, 7]
CARD = [0, 2, 4, 6]
NEED = 2
MAX_RESETS = 30
MAX_INNER = 120


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


def visible_monsters(fs):
    out_list = []
    for m in fs.get("monsters", []):
        pos = get_pos(m)
        if pos is not None and m.get("is_in_view", 0) == 1:
            out_list.append((pos[0], pos[1], m.get("monster_id", -1)))
    return out_list


def workflow(envs, agents, logger=None, monitor=None, *a, **k):
    out("PROBE v24 START")
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
            fs = obs["frame_state"]
            view = parse_view(obs.get("map_info"))
            cur = get_pos(fs["heroes"])

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
                dx, dz = DIRS[action]
                tgt_cell = (before[0] + dx, before[1] + dz)
                mons = visible_monsters(fs)
                out(f"DIAG act={action} at {before} target={tgt_cell} "
                    f"targetVal={tgt} e1={e1} e2={e2}")
                out(f"    monsters visible: {mons}")
                dump_3x3(view)
                at_tgt = [m for m in mons if (m[0], m[1]) == tgt_cell]
                near_tgt = [m for m in mons
                            if abs(m[0] - tgt_cell[0]) + abs(m[1] - tgt_cell[1]) <= 1]
                if at_tgt:
                    out(f"    *** MONSTER AT TARGET {tgt_cell}")
                elif near_tgt:
                    out(f"    *** MONSTER ADJACENT TO TARGET: {near_tgt}")
            env_reward, env_obs = env.step(action)
            if "observation" not in env_obs:
                out("no observation, reset")
                break
            after = get_pos(env_obs["observation"]["frame_state"]["heroes"])
            moved = (after[0] - before[0], after[1] - before[1])
            if corner_info is not None:
                good = moved == DIRS[action]
                counts[action][0 if good else 1] += 1
                if not good:
                    mons_after = visible_monsters(
                        env_obs["observation"]["frame_state"])
                    out(f"    BLOCKED moved={moved}; monsters after={mons_after}")
                    blocked_positions.add(before)
                else:
                    out(f"    OK moved={moved} counts={counts[action]}")
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
    out("PROBE v24 END")
    time.sleep(3)
