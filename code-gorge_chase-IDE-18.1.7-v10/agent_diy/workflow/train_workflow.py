#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Probe v10 (non-training) workflow for Gorge Chase.
峡谷追猎探针 v10（不训练，走 agent_diy 自定义算法）。

Fixes over v9 / 相对 v9 的修复:
  - Safe field access: never touch pos of out-of-view monsters (v9 crashed here).
    安全字段访问：绝不访问视野外怪物的 pos（v9 在此崩溃）。
  - Full traceback capture around every step.
    每步都包裹异常捕获并打印完整 traceback。
  - Build the map by stitching per-step vision map_info (no direct JSON needed).
    用每步视野 map_info 拼接地图（不依赖直接读 JSON）。
  - Survival policy: BFS distance field from visible monsters, wall-aware.
    存活策略：从可见怪物做 BFS 距离场，能利用障碍。
"""

import os
import math
import time
import traceback
from collections import deque

import numpy as np

from tools.train_env_conf_validate import read_usr_conf


# Action 0 right, 1 right-down, 2 down, 3 left-down, 4 left, 5 left-up, 6 up, 7 right-up.
# 8-neighbor deltas (x, z).
DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
GRID = 128


def get_pos(obj):
    """Return (x, z) from an object, or None if pos missing (e.g. out-of-view)."""
    if not isinstance(obj, dict):
        return None
    pos = obj.get("pos")
    if not isinstance(pos, dict):
        return None
    x, z = pos.get("x"), pos.get("z")
    if x is None or z is None:
        return None
    return int(x), int(z)


# ===================== Map discovery / 地图探查 =====================

def probe_map_files(logger):
    logger.info("===== PROBE v10 [1] filesystem map search =====")
    for r in ["/data/projects", "/data", "/workspace", "/app", "/data/projects/kaiwu_arena"]:
        try:
            logger.info(f"DIR {r} -> {os.listdir(r)}")
        except Exception as e:
            logger.info(f"DIR {r} failed: {repr(e)}")

    found = []
    for base in ["/data/projects", "/workspace", "/app"]:
        try:
            for dp, dn, fn in os.walk(base):
                if dp.count(os.sep) > 7:
                    dn[:] = []
                for f in fn:
                    if f.endswith(".json") and ("map" in f.lower() or "gorge" in f.lower()):
                        found.append(os.path.join(dp, f))
        except Exception as e:
            logger.info(f"WALK {base} failed: {repr(e)}")
    found = sorted(set(found))
    logger.info(f"map json found: {found}")
    for p in found[:5]:
        try:
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
            logger.info(f"===== MAP {p} bytes={len(content)} =====")
            for i in range(0, len(content), 6000):
                logger.info(f"MAPCHUNK {p} :: {content[i:i+6000]}")
        except Exception as e:
            logger.info(f"read {p} failed: {repr(e)}")


# ===================== Vision-based map stitching / 视野拼接地图 =====================

def merge_view(global_map, view, pos):
    """Merge a square vision view centered at pos into the global map (0=walkable,1=blocked)."""
    vs = view.shape[0]
    half = vs // 2
    px, pz = pos
    l = max(0, px - half)
    r = min(GRID, px + half + 1)
    u = max(0, pz - half)
    d = min(GRID, pz + half + 1)
    vl = l - (px - half)
    vr = vl + (r - l)
    vu = u - (pz - half)
    vd = vu + (d - u)
    crop = view[vl:vr, vu:vd]
    global_map[l:r, u:d] = crop


# ===================== BFS distance field / BFS 距离场 =====================

def bfs_distance(global_map, sources):
    INF = 9999
    dist = [[INF] * GRID for _ in range(GRID)]
    q = deque()
    for sx, sz in sources:
        if 0 <= sx < GRID and 0 <= sz < GRID and global_map[sx][sz] == 0:
            if dist[sx][sz] != 0:
                dist[sx][sz] = 0
                q.append((sx, sz))
    while q:
        x, z = q.popleft()
        base = dist[x][z]
        for dx, dz in DIRS:
            nx, nz = x + dx, z + dz
            if 0 <= nx < GRID and 0 <= nz < GRID and global_map[nx][nz] == 0 and dist[nx][nz] == INF:
                dist[nx][nz] = base + 1
                q.append((nx, nz))
    return dist


def pick_bfs_move(obs, global_map, monsters):
    """Pick move maximizing BFS path length from visible monsters; avoid dead ends."""
    legal = obs["legal_action"]
    hpos = get_pos(obs["frame_state"]["heroes"])
    if hpos is None:
        return 0
    hx, hz = hpos
    visible = [m for m in monsters if m.get("is_in_view", 0) == 1]
    sources = [p for m in visible for p in [get_pos(m)] if p is not None]
    dist = bfs_distance(global_map, sources) if sources else None

    best, bs = None, -1e18
    for d, (dx, dz) in enumerate(DIRS):
        if not legal[d]:
            continue
        nx, nz = hx + dx, hz + dz
        if not (0 <= nx < GRID and 0 <= nz < GRID):
            continue
        if global_map[nx][nz] != 0:
            continue
        dval = dist[nx][nz] if dist is not None else 100
        # Count open neighbors of the target cell to avoid running into dead ends.
        open_nbrs = 0
        for ddx, ddz in DIRS:
            ax, az = nx + ddx, nz + ddz
            if 0 <= ax < GRID and 0 <= az < GRID and global_map[ax][az] == 0:
                open_nbrs += 1
        score = dval + 0.5 * open_nbrs
        if score > bs:
            bs, best = score, d
    return best if best is not None else 0


# ===================== Episode probe / 单局探针 =====================

def probe_episode(envs, usr_conf, logger):
    logger.info("===== PROBE v10 [2] episode trace =====")
    env = envs[0]
    env_obs = env.reset(usr_conf)
    # 1 = unknown / blocked; vision sets 0 for walkable cells.
    global_map = np.ones((GRID, GRID), dtype=np.int32)
    step = 0
    step_times = []

    while True:
        try:
            obs = env_obs["observation"]
            fs = obs["frame_state"]
            hero = fs["heroes"]
            hpos = get_pos(hero)
            monsters = fs.get("monsters", [])
            organs = fs.get("organs", [])

            map_info = obs.get("map_info")
            if map_info and hpos is not None:
                try:
                    if isinstance(map_info[0], dict):
                        view = np.array([x["values"] for x in map_info], dtype=np.int32)
                    else:
                        view = np.array(map_info, dtype=np.int32)
                    merge_view(global_map, view, hpos)
                except Exception as e:
                    logger.info(f"map merge failed: {repr(e)}")

            minfo = [
                (m.get("monster_id", -1), get_pos(m), m.get("is_in_view", 0),
                 m.get("hero_l2_distance", -1))
                for m in monsters
            ]
            oinfo = [
                (o.get("sub_type", -1), get_pos(o), o.get("status", -1))
                for o in organs
            ]
            if step < 80 or step % 25 == 0:
                logger.info(
                    f"TRACE step={step} hero={hpos} buff_rem={hero.get('buff_remaining_time', 0)} "
                    f"flash_cd={hero.get('flash_cooldown', -1)} monsters={minfo} organs={oinfo}"
                )

            act = pick_bfs_move(obs, global_map, monsters)
            t0 = time.time()
            _r, env_obs = env.step(act)
            dt = time.time() - t0
            step_times.append(dt)
            step += 1
            if step <= 15 or step % 50 == 0:
                logger.info(f"STEPTIME step={step} act={act} env.step={dt*1000:.1f}ms")

            term = env_obs.get("terminated")
            trunc = env_obs.get("truncated")
            if term or trunc:
                logger.info(f"===== EPISODE done step={step} term={term} trunc={trunc} =====")
                break
            if step >= 1000:
                logger.info("===== reached 1000 steps (full survival) =====")
                break
        except Exception as e:
            logger.error(f"step loop crashed at step={step}: {repr(e)}")
            logger.error(traceback.format_exc())
            break

    if step_times:
        arr = sorted(step_times)
        n = len(arr)
        logger.info(
            f"===== PROBE v10 [3] step-time over {n}: mean={sum(arr)/n*1000:.1f}ms "
            f"p50={arr[n//2]*1000:.1f}ms max={arr[-1]*1000:.1f}ms ====="
        )
    known = int((global_map == 0).sum())
    logger.info(f"known walkable cells: {known} / {GRID*GRID}")


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    logger.info("########## PROBE v10 START ##########")
    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    try:
        probe_map_files(logger)
    except Exception as e:
        logger.error(f"probe_map_files crashed: {repr(e)}")
        logger.error(traceback.format_exc())

    try:
        probe_episode(envs, usr_conf, logger)
    except Exception as e:
        logger.error(f"probe_episode crashed: {repr(e)}")
        logger.error(traceback.format_exc())

    logger.info("########## PROBE v10 COMPLETE; you may RELEASE this task ##########")
    # Give the log pipeline time to flush.
    time.sleep(8)
    return
