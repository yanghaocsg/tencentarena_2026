#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""v20: settle the diagonal corner-cutting rule.

Scans the initial 21x21 view for diagonal test cases where:
  - the start cell is passable;
  - the diagonal target is passable;
  - EXACTLY ONE of the two adjacent edges is an obstacle (the other is
    passable).
Then it walks to each start cell (BFS within the known view), performs the
diagonal action, and reports whether the move actually happened (moved =
the diagonal delta) or was blocked (moved = (0,0)).

If every such case is BLOCKED, the rule is "both edges must be passable"
(no corner cutting). If they succeed, the documented "at least one" rule
holds.
"""
import time
from collections import deque

import numpy as np

from tools.train_env_conf_validate import read_usr_conf

GRID = 128
DIRS = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
DIAG = [1, 3, 5, 7]


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
    cases = []
    for r in range(21):
        for c in range(21):
            if view[r, c] != 1:
                continue
            for d in DIAG:
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
                    cases.append((x, z, d, v1, v2))
    return cases


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


def step_to(env, env_obs, act):
    env_reward, env_obs = env.step(act)
    obs = env_obs["observation"]
    new = get_pos(obs["frame_state"]["heroes"])
    return env_obs, new


def workflow(envs, agents, logger=None, monitor=None, *a, **k):
    out("PROBE v20 START")
    uc = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    env = envs[0]
    env_obs = env.reset(usr_conf=uc)
    obs = env_obs["observation"]
    h0 = get_pos(obs["frame_state"]["heroes"])
    view = parse_view(obs.get("map_info"))
    out(f"hero={h0}")

    cases = find_cases(view, h0[0], h0[1])
    out(f"found {len(cases)} corner-cutting test cases")
    for i, (x, z, d, v1, v2) in enumerate(cases):
        out(f"case{i}: start=({x},{z}) diagAct={d} edgeVals=({v1},{v2})")

    cur = h0
    n_blocked = 0
    n_ok = 0
    for i, (x, z, d, v1, v2) in enumerate(cases):
        # walk to start cell
        sr, sc = cur[0] - (h0[0] - 10), cur[1] - (h0[1] - 10)
        gr, gc = x - (h0[0] - 10), z - (h0[1] - 10)
        path = bfs_path(view, sr, sc, gr, gc)
        if path is None:
            out(f"case{i}: no path to start, skip")
            continue
        for (pr, pc) in path:
            dx = pr - (cur[0] - (h0[0] - 10))
            dz = pc - (cur[1] - (h0[1] - 10))
            act = DIRS.index((dx, dz))
            env_obs, cur = step_to(env, env_obs, act)
        # perform diagonal
        before = cur
        env_obs, after = step_to(env, env_obs, d)
        moved = (after[0] - before[0], after[1] - before[1])
        expect = DIRS[d]
        result = "OK" if moved == expect else "BLOCKED"
        if moved == expect:
            n_ok += 1
        else:
            n_blocked += 1
        out(f"case{i}: before={before} after={after} moved={moved} "
            f"expect={expect} -> {result}")
        cur = after

    out(f"SUMMARY: OK={n_ok} BLOCKED={n_blocked}")
    if n_ok == 0 and n_blocked > 0:
        out("CONCLUSION: diagonal needs BOTH edges passable (no corner cutting)")
    elif n_blocked == 0 and n_ok > 0:
        out("CONCLUSION: diagonal needs only ONE edge (documented rule)")
    out("PROBE v20 END")
    time.sleep(3)
