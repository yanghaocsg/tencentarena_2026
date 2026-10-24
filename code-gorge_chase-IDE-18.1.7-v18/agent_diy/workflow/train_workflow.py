#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""v18: BFS survival with monster memory + stuck recovery.

Fixes two v17 failure modes:
1. When the monster leaves view, BFS had no source and the distance field
   became all-INF, so the picker always returned d=0 (straight right) into
   a wall. Now we remember the LAST KNOWN monster cell(s) and keep using
   them as BFS sources while the monster is out of view.
2. When the hero stopped moving (v17 stayed on one cell for 37 steps),
   it kept requesting the same action. Now we detect a stuck hero (same
   cell N steps) and rotate the tie-breaking start direction, and print
   a full diagnostic (8 neighbours, legal action, raw view) on the first
   stuck event.
"""
import time
from collections import deque

import numpy as np

from tools.train_env_conf_validate import read_usr_conf

GRID = 128
PASSABLE, BLOCKED, UNKNOWN = 1, 0, -1
DIRS = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
STUCK_LIMIT = 3


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


def merge_view(gmap, view, pos):
    vs = view.shape[0]
    half = vs // 2
    px, pz = pos
    l = max(0, px - half)
    r = min(GRID, px + half + 1)
    u = max(0, pz - half)
    d = min(GRID, pz + half + 1)
    vl = l - (px - half)
    vu = u - (pz - half)
    gmap[l:r, u:d] = view[vl:vl + (r - l), vu:vu + (d - u)]


def bfs_distance(gmap, sources, limit):
    INF = 99999
    dist = np.full((GRID, GRID), INF, dtype=np.int32)
    q = deque()
    seen = set()
    for sx, sz in sources:
        if 0 <= sx < GRID and 0 <= sz < GRID and gmap[sx, sz] == PASSABLE:
            if (sx, sz) not in seen:
                seen.add((sx, sz))
                dist[sx, sz] = 0
                q.append((sx, sz))
    while q:
        x, z = q.popleft()
        nd = dist[x, z] + 1
        if nd > limit:
            continue
        for dx, dz in DIRS:
            nx, nz = x + dx, z + dz
            if 0 <= nx < GRID and 0 <= nz < GRID \
                    and gmap[nx, nz] == PASSABLE and dist[nx, nz] > nd:
                dist[nx, nz] = nd
                q.append((nx, nz))
    return dist


def neighbour_report(gmap, hpos, legal):
    lines = []
    for d in range(8):
        dx, dz = DIRS[d]
        nx, nz = hpos[0] + dx, hpos[1] + dz
        if 0 <= nx < GRID and 0 <= nz < GRID:
            lines.append(f"d{d}->({nx},{nz})={gmap[nx,nz]} legal={legal[d]}")
        else:
            lines.append(f"d{d}->OOB legal={legal[d]}")
    return " | ".join(lines)


def pick_move(gmap, legal, hpos, sources, start_dir):
    dist = bfs_distance(gmap, sources, limit=120)
    order = [(start_dir + i) % 8 for i in range(8)]
    best, bestval = None, -1
    for d in order:
        if not legal[d]:
            continue
        dx, dz = DIRS[d]
        nx, nz = hpos[0] + dx, hpos[1] + dz
        if not (0 <= nx < GRID and 0 <= nz < GRID):
            continue
        if gmap[nx, nz] != PASSABLE:
            continue
        v = int(dist[nx, nz])
        if v > bestval:
            bestval, best = v, d
    if best is not None:
        return best, "dist"
    for d in order:
        if not legal[d]:
            continue
        dx, dz = DIRS[d]
        nx, nz = hpos[0] + dx, hpos[1] + dz
        if 0 <= nx < GRID and 0 <= nz < GRID and gmap[nx, nz] == UNKNOWN:
            return d, "unknown"
    for d in order:
        if legal[d]:
            return d, "legal"
    return 0, "zero"


def save_map(gmap, path):
    with open(path, "w") as f:
        for x in range(GRID):
            row = []
            for z in range(GRID):
                v = gmap[x, z]
                row.append("#" if v == BLOCKED else ("." if v == PASSABLE else "?"))
            f.write("".join(row) + "\n")


def probe_episode(env, usr_conf, ei):
    env_obs = env.reset(usr_conf)
    gmap = np.full((GRID, GRID), UNKNOWN, dtype=np.int32)
    step = 0
    t0 = time.time()
    last_mon = []
    prev_pos = None
    stuck_count = 0
    start_dir = 0
    diag_done = False
    while True:
        obs = env_obs["observation"]
        fs = obs["frame_state"]
        hero = fs["heroes"]
        hpos = get_pos(hero)
        monsters_raw = fs.get("monsters", [])
        organs = fs.get("organs", [])
        cur_mon = []
        for m in monsters_raw:
            mp = get_pos(m)
            if mp is not None:
                cur_mon.append(mp)
        if cur_mon:
            last_mon = cur_mon
        sources = cur_mon if cur_mon else last_mon

        view = parse_view(obs.get("map_info"))
        if view is not None and hpos is not None:
            merge_view(gmap, view, hpos)
        legal = obs.get("legal_action") or [0] * 16

        # stuck detection
        if hpos == prev_pos:
            stuck_count += 1
        else:
            stuck_count = 0
        if stuck_count >= STUCK_LIMIT:
            start_dir = (start_dir + 1) % 8
            if not diag_done:
                out(f"STUCK e{ei} s={step} hero={hpos} sources={sources} "
                    f"start_dir={start_dir}")
                out(f"  legal={list(legal)}")
                out(f"  {neighbour_report(gmap, hpos, legal)}")
                if view is not None:
                    out(f"  raw view centerROW={view[10].tolist()}")
                diag_done = True
            stuck_count = 0
        prev_pos = hpos

        act, why = pick_move(gmap, legal, hpos, sources, start_dir)

        if step < 30 or step % 50 == 0:
            buff = hero.get("buff_remaining_time", 0)
            cd = hero.get("flash_cooldown", 0)
            out(f"TRACE e{ei} s={step} hero={hpos} act={act}({why}) "
                f"cur={cur_mon} last={last_mon} buff={buff} cd={cd} "
                f"org={len(organs)} sd={start_dir}")

        env_reward, env_obs = env.step(act)
        step += 1

        if env_obs.get("terminated") or env_obs.get("truncated"):
            out(f"EPISODE e{ei} END s={step} term={env_obs.get('terminated')} "
                f"trunc={env_obs.get('truncated')} elapsed={time.time()-t0:.1f}s")
            save_map(gmap, f"/tmp/gmap_e{ei}.txt")
            out(f"MAP e{ei} passable={int((gmap==PASSABLE).sum())} "
                f"blocked={int((gmap==BLOCKED).sum())} "
                f"unknown={int((gmap==UNKNOWN).sum())}")
            return
        if step > 1000:
            out(f"EPISODE e{ei} REACHED 1000 (WIN)")
            save_map(gmap, f"/tmp/gmap_e{ei}.txt")
            return


def workflow(envs, agents, logger=None, monitor=None, *a, **k):
    out("PROBE v18 START")
    uc = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    for ei in range(len(envs)):
        try:
            probe_episode(envs[ei], uc, ei)
        except Exception as e:  # noqa
            import traceback
            out("CRASH:\n" + traceback.format_exc())
    out("PROBE v18 END")
    time.sleep(3)
