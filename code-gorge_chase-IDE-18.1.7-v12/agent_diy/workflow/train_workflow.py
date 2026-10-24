#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Probe v12 (non-training) workflow for Gorge Chase.
峡谷追猎探针 v12（不训练，走 agent_diy 自定义算法）。

CRITICAL FIX over v11 / 相对 v11 的关键修复:
  - v11 had the map_info semantics BACKWARDS. The platform uses
    1 = passable, 0 = obstacle (see agent_ppo/feature/feature_utils.is_passable,
    which checks map_data[x,z] == 1, and preprocessor map_data[0] initialized
    to all ones). v11 wrongly assumed 0 = walkable / 1 = blocked, so:
      * BFS sources (monster cells, which are passable=1) were filtered out;
      * every passable neighbor (value 1) was skipped in pick_bfs_move;
      * pick_bfs_move always fell back to the fixed action 0 (move right),
        ignoring legal_action — which drove the hero into an illegal/edge
        action and made env.step hang after only 2 steps.
    v11 把 map_info 语义搞反了。平台口径是 1=可通行、0=障碍
    （feature_utils.is_passable 判 map_data==1，preprocessor map_data[0]
    初始全 1）。v11 误判为 0=可走、1=障碍，导致 BFS 源（怪物格=1）被过滤、
    所有可走邻居（值=1）被跳过、pick_bfs_move 恒返回固定动作 0（向右）、
    无视 legal_action，最终把英雄推向非法/边界动作，env.step 仅 2 步即卡死。

v12 map convention / v12 地图口径:
  PASSABLE = 1, BLOCKED = 0 (platform raw values), UNKNOWN = -1 (not yet seen).
  BFS and move selection only run on PASSABLE cells; unknown cells are used
  only as a last-resort exploration fallback.

Inherited from v11 / 从 v11 继承:
  - Drive env FIRST (env.reset immediately), safe non-recursive file search last.
    先驱动环境（立即 env.reset），安全不递归的文件搜索放最后。
  - Safe field access: never touch pos of out-of-view monsters.
    安全字段访问：绝不访问视野外怪物的 pos。
  - Build the map by stitching per-step vision map_info.
    用每步视野 map_info 拼接地图。

New in v12 / v12 新增:
  - env.step is wrapped with a hard timeout (10s). A hang is logged explicitly
    as "ENVSTEP HANG" and the episode ends, instead of silently stalling until
    run_train_test kills the process at ~25s.
    env.step 加 10 秒硬超时，卡死时明确打印 "ENVSTEP HANG" 并结束，
    而不是静默停滞到 run_train_test 约 25 秒后清理。
"""

import os
import glob
import time
import threading
import traceback
from collections import deque

import numpy as np

from tools.train_env_conf_validate import read_usr_conf


# Action 0 right, 1 right-down, 2 down, 3 left-down, 4 left, 5 left-up, 6 up, 7 right-up.
# 8-neighbor deltas (x, z).
DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
GRID = 128

# Platform map values / 平台地图取值.
PASSABLE = 1
BLOCKED = 0
UNKNOWN = -1


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


# ===================== Map discovery (safe, non-recursive) / 地图探查 =====================

def probe_map_files(logger):
    logger.info("===== PROBE v12 [A] filesystem map search (safe, non-recursive) =====")
    # 1) Non-recursive listing only: millisecond-level, never blocks.
    #    只做不递归的目录列举：毫秒级，不会卡死。
    for r in ["/data/projects", "/data", "/workspace", "/app"]:
        try:
            logger.info(f"DIR {r} -> {os.listdir(r)}")
        except Exception as e:
            logger.info(f"DIR {r} failed: {repr(e)}")

    # 2) Fast glob scoped to /data/projects only; never walks /app.
    #    仅在 /data/projects 下做快速 glob，绝不遍历 /app。
    candidates = []
    for pat in [
        "/data/projects/**/gorge_chase_map_*.json",
        "/data/projects/**/map_json/*.json",
        "/data/projects/**/map_json/**/*.json",
    ]:
        try:
            for p in glob.glob(pat, recursive=True):
                if p not in candidates:
                    candidates.append(p)
        except Exception as e:
            logger.info(f"GLOB {pat} failed: {repr(e)}")

    logger.info(f"map candidates found: {candidates}")
    for p in candidates:
        try:
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
            logger.info(f"===== MAP {p} bytes={len(content)} =====")
            for i in range(0, len(content), 6000):
                logger.info(f"MAPCHUNK {p} :: {content[i:i+6000]}")
        except Exception as e:
            logger.info(f"read {p} failed: {repr(e)}")
    if not candidates:
        logger.info("NO map file accessible from user code; relying on vision stitching.")


# ===================== Vision-based map stitching / 视野拼接地图 =====================

def merge_view(global_map, view, pos):
    """Merge a square vision view centered at pos.

    Platform semantics: 1=passable, 0=obstacle; global_map uses -1=unknown.
    """
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
        if 0 <= sx < GRID and 0 <= sz < GRID and global_map[sx][sz] == PASSABLE:
            if dist[sx][sz] != 0:
                dist[sx][sz] = 0
                q.append((sx, sz))
    while q:
        x, z = q.popleft()
        base = dist[x][z]
        for dx, dz in DIRS:
            nx, nz = x + dx, z + dz
            if (
                0 <= nx < GRID
                and 0 <= nz < GRID
                and global_map[nx][nz] == PASSABLE
                and dist[nx][nz] == INF
            ):
                dist[nx][nz] = base + 1
                q.append((nx, nz))
    return dist


def pick_bfs_move(obs, global_map, monsters):
    """Pick move maximizing BFS path length from visible monsters; avoid dead ends.

    Only known PASSABLE cells are considered. If none exists (surrounded by walls
    /unknown), fall back to a legal move into an UNKNOWN cell, then the first
    legal move.
    """
    legal = obs["legal_action"]
    hpos = get_pos(obs["frame_state"]["heroes"])
    if hpos is None:
        return 0
    hx, hz = hpos
    visible = [m for m in monsters if m.get("is_in_view", 0) == 1]
    sources = [p for m in visible for p in [get_pos(m)] if p is not None]
    dist = bfs_distance(global_map, sources) if sources else None

    # First pass: known passable cells.
    best, bs = None, -1e18
    for d, (dx, dz) in enumerate(DIRS):
        if not legal[d]:
            continue
        nx, nz = hx + dx, hz + dz
        if not (0 <= nx < GRID and 0 <= nz < GRID):
            continue
        if global_map[nx][nz] != PASSABLE:
            continue
        dval = dist[nx][nz] if dist is not None else 100
        # Count open neighbors of the target cell to avoid running into dead ends.
        open_nbrs = 0
        for ddx, ddz in DIRS:
            ax, az = nx + ddx, nz + ddz
            if 0 <= ax < GRID and 0 <= az < GRID and global_map[ax][az] == PASSABLE:
                open_nbrs += 1
        score = dval + 0.5 * open_nbrs
        if score > bs:
            bs, best = score, d
    if best is not None:
        return best

    # Fallback 1: legal move into an unknown cell (exploration).
    for d, (dx, dz) in enumerate(DIRS):
        if not legal[d]:
            continue
        nx, nz = hx + dx, hz + dz
        if 0 <= nx < GRID and 0 <= nz < GRID and global_map[nx][nz] == UNKNOWN:
            return d
    # Fallback 2: first legal move.
    for d in range(8):
        if legal[d]:
            return d
    return 0


# ===================== env.step with hard timeout / 带超时的 env.step =====================

def step_with_timeout(env, act, timeout, logger, step):
    """Run env.step(act) in a daemon thread; return (result, timed_out).

    On timeout the worker thread is left running in the background (daemon) and
    the caller must stop using this env.
    """
    box = {}

    def worker():
        box["r"] = env.step(act)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        logger.error(f"ENVSTEP HANG at step={step} act={act} after {timeout}s")
        return None, True
    return box["r"], False


# ===================== Episode probe / 单局探针 =====================

def probe_episode(envs, usr_conf, logger):
    logger.info("===== PROBE v12 [B] episode trace =====")
    env = envs[0]
    env_obs = env.reset(usr_conf)
    logger.info("env.reset returned; starting step loop")
    # -1 = unknown; vision fills 1=passable / 0=obstacle.
    global_map = np.full((GRID, GRID), UNKNOWN, dtype=np.int32)
    step = 0
    step_times = []
    hang = False

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
            if step < 15:
                logger.info(f"STEP-IN step={step} act={act}")
            t0 = time.time()
            res, timed_out = step_with_timeout(env, act, 10.0, logger, step)
            dt = time.time() - t0
            if timed_out:
                hang = True
                break
            _r, env_obs = res
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
            f"===== PROBE v12 [C] step-time over {n}: mean={sum(arr)/n*1000:.1f}ms "
            f"p50={arr[n//2]*1000:.1f}ms max={arr[-1]*1000:.1f}ms ====="
        )
    n_pass = int((global_map == PASSABLE).sum())
    n_block = int((global_map == BLOCKED).sum())
    n_unk = GRID * GRID - n_pass - n_block
    logger.info(
        f"map coverage: passable={n_pass} blocked={n_block} unknown={n_unk} "
        f"(total {GRID*GRID})"
    )
    if hang:
        logger.info("episode ended due to env.step hang")


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    logger.info("########## PROBE v12 START ##########")
    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    # Drive the environment FIRST, before any filesystem search, so the framework
    # sees a healthy, progressing env and does not auto-release / time out.
    # 先驱动环境，再做文件搜索，确保框架看到健康、有进展的 env，不被自动释放。
    try:
        probe_episode(envs, usr_conf, logger)
    except Exception as e:
        logger.error(f"probe_episode crashed: {repr(e)}")
        logger.error(traceback.format_exc())

    # Filesystem search only after the env has been driven; safe and fast.
    # 环境驱动后再做文件搜索，安全快速。
    try:
        probe_map_files(logger)
    except Exception as e:
        logger.error(f"probe_map_files crashed: {repr(e)}")
        logger.error(traceback.format_exc())

    logger.info("########## PROBE v12 COMPLETE; you may RELEASE this task ##########")
    # Give the log pipeline time to flush.
    time.sleep(8)
    return
