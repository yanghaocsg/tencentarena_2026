#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""v21: complete corner-cutting check across all 4 diagonal directions.

v20 proved DR and UL succeed with only one edge passable. This version
tests UR(1), UL(3), DL(5), DR(7) with two cases each, and is robust to a
dropped connection / episode end: every step result is checked, and a
missing 'observation' or terminated/truncated triggers a fresh reset
instead of a KeyError.
"""
import time
from collections import deque

import numpy as np

from tools.train_env_conf_validate import read_usr_conf

GRID = 128
DIRS = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
DIAG = [1, 3, 5, 7]
PER_DIR = 2


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


def find_cases(view, hx, hz):
    by_dir = {1: [], 3: [], 5: [], 7: []}
    for r in range(21):
        for c in range(21):
            if view[r, c] != 1:
                continue
            for d in DIAG:
                if len(by_dir[d]) >= PER_DIR:
                    continue
                dx, dz = DIRS[d]
                tr, tc = r + dx, c + dz
                if not (0 <= tr < 21 and 0 <= tc < 21):
                    continue
                if view[tr, tc] != 1:
                    continue
                e1r, e1c = r + dx, c
                e2r, e2c = r, c + dz
                v1 = int(view[e1r, e1c]) if 0 <= e1r < 21 and 0 <= e1c < 21 else -1
                v2 = int(view[e2r, e2c]) if 0 <= e2r < 21 and 0 <= e2c < 21 else -1
                if (v1 == 0 and v2 == 1) or (v1 == 1 and v2 == 0):
                    x, z = hx + r - 10, hz + c - 10
                    by_dir[d].append((x, z, d, v1, v2))
    selected = []
    for d in DIAG:
        selected.extend(by_dir[d])
    return selected


def bfs_path(view, sr, sc, gr, gc):
    q = deque([(sr, sc)])
    prev = {(sr, sc): None}
    while q:
        cur = q.popleft()
        if cur == (gr, gc):
            cells = []
            while cur != (sr, sc):
                cells.append(cur)
                cur = prev[cur]
            cells.reverse()
            return cells
        for d in range(8):
            dx, dz = DIRS[d]
            nr, nc = cur[0] + dx, cur[1] + dz
            if 0 <= nr < 21 and 0 <= nc < 21 and view[nr, nc] == 1 \
                    and (nr, nc) not in prev:
                prev[(nr, nc)] = cur
                q.append((nr, nc))
    return None


def do_step(env, env_obs, act, uc):
    """Return (env_obs, pos, reset_happened)."""
    env_reward, env_obs = env.step(act)
    if "observation" not in env_obs:
        env_obs = env.reset(usr_conf=uc)
        return env_obs, None, True
    pos = get_pos(env_obs["observation"]["frame_state"]["heroes"])
    if env_obs.get("terminated") or env_obs.get("truncated"):
        env_obs = env.reset(usr_conf=uc)
        return env_obs, pos, True
    return env_obs, pos, False


def workflow(envs, agents, logger=None, monitor=None, *a, **k):
    out("PROBE v21 START")
    uc = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    env = envs[0]
    env_obs = env.reset(usr_conf=uc)
    obs = env_obs["observation"]
    h0 = get_pos(obs["frame_state"]["heroes"])
    view = parse_view(obs.get("map_info"))
    out(f"hero={h0}")

    cases = find_cases(view, h0[0], h0[1])
    out(f"selected {len(cases)} cases (2 per diagonal direction)")
    for (x, z, d, v1, v2) in cases:
        out(f"  start=({x},{z}) act={d} edges=({v1},{v2})")

    cur = h0
    counts = {1: [0, 0], 3: [0, 0], 5: [0, 0], 7: [0, 0]}
    for (x, z, d, v1, v2) in cases:
        # After a reset the static view offsets change; re-derive from h0.
        sr = cur[0] - (h0[0] - 10)
        sc = cur[1] - (h0[1] - 10)
        gr = x - (h0[0] - 10)
        gc = z - (h0[1] - 10)
        path = bfs_path(view, sr, sc, gr, gc)
        if path is None:
            out(f"act={d}: no path, skip")
            continue
        ok_walk = True
        for (pr, pc) in path:
            dx = pr - (cur[0] - (h0[0] - 10))
            dz = pc - (cur[1] - (h0[1] - 10))
            try:
                act = DIRS.index((dx, dz))
            except ValueError:
                ok_walk = False
                break
            env_obs, cur, reset_h = do_step(env, env_obs, act, uc)
            if reset_h:
                cur = h0
                ok_walk = False
                break
        if not ok_walk:
            out(f"act={d}: walk interrupted, skip")
            continue
        before = cur
        env_obs, after, reset_h = do_step(env, env_obs, d, uc)
        if reset_h or after is None:
            out(f"act={d}: diagonal step reset, skip")
            cur = h0
            continue
        moved = (after[0] - before[0], after[1] - before[1])
        good = moved == DIRS[d]
        if good:
            counts[d][0] += 1
        else:
            counts[d][1] += 1
        out(f"act={d}: {before}->{after} moved={moved} expect={DIRS[d]} "
            f"{'OK' if good else 'BLOCKED'}")
        cur = after

    out("SUMMARY per direction [OK, BLOCKED]:")
    for d in DIAG:
        out(f"  act={d}: {counts[d]}")
    total_ok = sum(counts[d][0] for d in DIAG)
    total_bl = sum(counts[d][1] for d in DIAG)
    out(f"TOTAL OK={total_ok} BLOCKED={total_bl}")
    if total_bl == 0 and total_ok > 0:
        out("CONCLUSION: all diagonals need only ONE edge (documented rule)")
    out("PROBE v21 END")
    time.sleep(3)
