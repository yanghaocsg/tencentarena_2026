#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""v14: BFS survival probe. Correct access order (frame_state FIRST, then map_info).

Key fix vs v12: the raw obs is a lazy/shared object. You MUST touch
frame_state (heroes/monsters/organs) before iterating map_info, otherwise
the iteration silently blocks around row 4.
"""
import time
from collections import deque

import numpy as np

from tools.train_env_conf_validate import read_usr_conf

GRID = 128
VIEW_R = 10
PASSABLE, BLOCKED, UNKNOWN = 1, 0, -1

# Documented action order (18.1.7):
# 0 right, 1 upper-right, 2 up, 3 upper-left, 4 left,
# 5 lower-left, 6 down, 7 lower-right ; 8-15 flashes.
# x right positive, z DOWN positive (screen).
DIRS = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
FLASH_LEN = [10, 8, 10, 8, 10, 8, 10, 8]


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
    for sx, sz in sources:
        if 0 <= sx < GRID and 0 <= sz < GRID and gmap[sx, sz] == PASSABLE:
            if dist[sx, sz] != 0:
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


def pick_move(gmap, legal, hpos, monsters):
    dist = bfs_distance(gmap, monsters, limit=120)
    best, bestval = None, -1
    for d in range(8):
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
    for d in range(8):
        if not legal[d]:
            continue
        dx, dz = DIRS[d]
        nx, nz = hpos[0] + dx, hpos[1] + dz
        if 0 <= nx < GRID and 0 <= nz < GRID and gmap[nx, nz] == UNKNOWN:
            return d, "unknown"
    for d in range(8):
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


def probe_episode(envs, usr_conf, ei, logger):
    env = envs[ei]
    env_obs = env.reset(usr_conf)
    gmap = np.full((GRID, GRID), UNKNOWN, dtype=np.int32)
    step = 0
    t0 = time.time()
    max_dt = 0.0
    while True:
        obs = env_obs["observation"]
        fs = obs["frame_state"]
        # frame_state MUST be touched before map_info.
        hero = fs["heroes"]
        hpos = get_pos(hero)
        monsters_raw = fs.get("monsters", [])
        organs = fs.get("organs", [])
        monsters = []
        for m in monsters_raw:
            mp = get_pos(m)
            if mp is not None:
                monsters.append(mp)
        view = parse_view(obs.get("map_info"))
        if view is not None and hpos is not None:
            merge_view(gmap, view, hpos)
        legal = obs.get("legal_action") or [0] * 16
        act, why = pick_move(gmap, legal, hpos, monsters)

        if step < 30 or step % 50 == 0:
            buff = hero.get("buff_remaining_time", 0)
            cd = hero.get("flash_cooldown", 0)
            logger.info(
                f"TRACE e{ei} s={step} hero={hpos} act={act}({why}) "
                f"mon={monsters} buff={buff} cd={cd} org={len(organs)}")

        t1 = time.time()
        env_reward, env_obs = env.step(act)
        dt = (time.time() - t1) * 1000.0
        max_dt = max(max_dt, dt)
        if step < 8:
            logger.info(f"STEPTIME e{ei} s={step} dt={dt:.1f}ms")
        step += 1

        if env_obs.get("terminated") or env_obs.get("truncated"):
            logger.info(
                f"EPISODE e{ei} END s={step} term={env_obs.get('terminated')} "
                f"trunc={env_obs.get('truncated')} maxdt={max_dt:.1f}ms "
                f"elapsed={time.time()-t0:.1f}s")
            save_map(gmap, f"/tmp/gmap_e{ei}.txt")
            logger.info(f"MAP e{ei} saved passable={int((gmap==PASSABLE).sum())} "
                        f"blocked={int((gmap==BLOCKED).sum())} "
                        f"unknown={int((gmap==UNKNOWN).sum())}")
            return
        if step > 1000:
            return


def workflow(envs, agents, logger=None, monitor=None, *a, **k):
    logger.info("PROBE v14 START")
    uc = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    for ei in range(len(envs)):
        try:
            probe_episode(envs, uc, ei, logger)
        except Exception as e:
            logger.error(f"PROBE e{ei} crash: {repr(e)}")
            import traceback
            logger.error(traceback.format_exc())
    logger.info("PROBE v14 END")
    time.sleep(5)
