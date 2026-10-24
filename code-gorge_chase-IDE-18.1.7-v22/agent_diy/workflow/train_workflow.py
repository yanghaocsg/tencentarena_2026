#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""v22: local corner-cutting check (no long-distance walks).

At each step the hero inspects its own 4 diagonal moves. If a diagonal
target is passable and exactly one adjacent edge is blocked (the other
open), it performs that diagonal and records whether the move actually
happened. When no such corner exists locally, it takes one exploratory
cardinal step. On episode end it resets and rescans. Each diagonal
direction needs 2 recorded results.

This avoids the v21 bug of walking 10+ cells with stale cross-reset
coordinates.
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


def local_corner(view, d):
    dx, dz = DIRS[d]
    tr, tc = 10 + dx, 10 + dz
    if not (0 <= tr < 21 and 0 <= tc < 21):
        return None
    tgt = int(view[tr, tc])
    e1 = int(view[10 + dx, 10])
    e2 = int(view[10, 10 + dz])
    if tgt == 1 and ((e1 == 0 and e2 == 1) or (e1 == 1 and e2 == 0)):
        return e1, e2
    return None


def raw_step(env, act):
    return env.step(act)


def workflow(envs, agents, logger=None, monitor=None, *a, **k):
    out("PROBE v22 START")
    uc = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    env = envs[0]
    counts = {1: [0, 0], 3: [0, 0], 5: [0, 0], 7: [0, 0]}
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
            for d in DIAG:
                if counts[d][0] + counts[d][1] >= NEED:
                    continue
                if local_corner(view, d) is not None:
                    action = d
                    break
            if action is None:
                # explore one cardinal step, rotating preference
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
            env_reward, env_obs = raw_step(env, action)
            if "observation" not in env_obs:
                out(f"step act={action} returned no observation, reset")
                break
            after = get_pos(env_obs["observation"]["frame_state"]["heroes"])
            moved = (after[0] - before[0], after[1] - before[1])
            if action in DIAG:
                good = moved == DIRS[action]
                counts[action][0 if good else 1] += 1
                out(f"act={action} {before}->{after} moved={moved} "
                    f"expect={DIRS[action]} {'OK' if good else 'BLOCKED'} "
                    f"counts={counts[action]}")
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
    if total_bl == 0 and total_ok > 0:
        out("CONCLUSION: diagonal needs only ONE edge open (documented rule)")
    elif total_ok > 0:
        out("NOTE: mixed; inspect individual cases")
    out("PROBE v22 END")
    time.sleep(3)
