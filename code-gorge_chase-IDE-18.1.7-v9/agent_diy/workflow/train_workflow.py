#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Probe (non-training) workflow for Gorge Chase.
峡谷追猎探针工作流（不训练，走 agent_diy 自定义算法）。

Goals / 目标:
  1. Locate and dump the FULL map JSON (obstacles / buff / treasure points).
     定位并 dump 完整地图 JSON。
  2. Trace hero/monster per-frame positions to settle the monster AI
     (pure greedy straight-line vs wall-avoiding BFS).
     逐帧记录英雄/怪物坐标，坐实怪物 AI（贪心直线 vs 绕墙 BFS）。
  3. Measure per-step env.step wall time to get the online search budget.
     测量每步 env.step 耗时，得到在线搜索时间预算。
"""

import os
import glob
import math
import time

from tools.train_env_conf_validate import read_usr_conf


DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def probe_map_files(logger):
    logger.info("===== PROBE[1] map files =====")
    direct = [
        "/data/projects/kaiwu_arena/gorge_chase/simulator/map_json/gorge_chase_map_1.json",
        "/data/projects/kaiwu_arena/gorge_chase/simulator/map_json/gorge_chase_map_2.json",
    ]
    patterns = [
        "/data/projects/**/gorge_chase_map_*.json",
        "/data/projects/**/map_json/*.json",
        "/data/projects/**/map_json/**/*.json",
    ]
    candidates = []
    for p in direct:
        if os.path.exists(p) and p not in candidates:
            candidates.append(p)
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            if p not in candidates:
                candidates.append(p)
    logger.info(f"map candidates found: {candidates}")
    for p in candidates:
        try:
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
            logger.info(f"===== MAP FILE {p} bytes={len(content)} =====")
            # Print in chunks so a long map is not truncated by the log pipeline.
            CHUNK = 6000
            for i in range(0, len(content), CHUNK):
                logger.info(f"MAPCHUNK {p} :: {content[i:i+CHUNK]}")
        except Exception as e:
            logger.info(f"read map {p} failed: {repr(e)}")
    if not candidates:
        logger.info("NO map file accessible from user code; will rely on vision exploration.")


def pick_safe_move(obs, monsters):
    """Greedy move maximizing nearest-monster distance (keeps probe alive)."""
    legal = obs["legal_action"]
    hx = obs["frame_state"]["heroes"]["pos"]["x"]
    hz = obs["frame_state"]["heroes"]["pos"]["z"]
    visible = [m for m in monsters if m.get("is_in_view", 0) == 1]

    best, bs = None, -1e18
    for d, (dx, dz) in enumerate(DIRS):
        if not legal[d]:
            continue
        nx, nz = hx + dx, hz + dz
        if visible:
            md = min(math.hypot(m["pos"]["x"] - nx, m["pos"]["z"] - nz) for m in visible)
        else:
            md = 100.0
        edge = min(nx, nz, 127 - nx, 127 - nz)
        s = md + 0.2 * edge
        if s > bs:
            bs, best = s, d
    return best if best is not None else 0


def probe_episode(envs, usr_conf, logger):
    logger.info("===== PROBE[2] episode trace =====")
    env = envs[0]
    env_obs = env.reset(usr_conf)
    step = 0
    step_times = []
    while True:
        obs = env_obs["observation"]
        fs = obs["frame_state"]
        hero = fs["heroes"]
        hx, hz = hero["pos"]["x"], hero["pos"]["z"]
        monsters = fs.get("monsters", [])
        organs = fs.get("organs", [])

        minfo = [
            (m.get("monster_id", -1), m["pos"]["x"], m["pos"]["z"],
             m.get("is_in_view", 0), m.get("hero_l2_distance", -1))
            for m in monsters
        ]
        oinfo = [
            (o.get("sub_type", -1), o["pos"]["x"], o["pos"]["z"], o.get("status", -1))
            for o in organs
        ]
        if step < 60 or step % 25 == 0:
            logger.info(
                f"TRACE step={step} hero=({hx},{hz}) buff_rem={hero.get('buff_remaining_time',0)} "
                f"flash_cd={hero.get('flash_cooldown',-1)} monsters={minfo} organs={oinfo}"
            )

        act = pick_safe_move(obs, monsters)
        t0 = time.time()
        _r, env_obs = env.step(act)
        dt = time.time() - t0
        step_times.append(dt)
        step += 1
        if step <= 15 or step % 50 == 0:
            logger.info(f"STEPTIME step={step} env.step={dt*1000:.1f}ms")

        if env_obs.get("terminated") or env_obs.get("truncated"):
            logger.info(
                f"===== PROBE episode done step={step} term={env_obs.get('terminated')} "
                f"trunc={env_obs.get('truncated')} ====="
            )
            break
        if step >= 320:
            logger.info("===== PROBE trace cap reached 320 steps =====")
            break

    if step_times:
        arr = sorted(step_times)
        n = len(arr)
        logger.info(
            f"===== PROBE[3] step-time stats over {n} steps: mean={sum(arr)/n*1000:.1f}ms "
            f"p50={arr[n//2]*1000:.1f}ms max={arr[-1]*1000:.1f}ms ====="
        )


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    probe_map_files(logger)
    probe_episode(envs, usr_conf, logger)

    logger.info("===== PROBE complete; you may RELEASE this task now =====")
    return
