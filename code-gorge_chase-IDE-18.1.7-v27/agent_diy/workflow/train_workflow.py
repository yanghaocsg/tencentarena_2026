#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""v27: diagnostic - dump full monster fields incl. out-of-view signals.

Goal: learn the exact format of hero_l2_distance and hero_relative_direction
for monsters that are OUT of view (pos = (-1,-1)), so the planner can infer
their real position instead of trusting a stale last-known cell.

Strategy: walk to explore (cardinal moves), and every step print each
monster's full dict. Runs 2 episodes.
"""
import time

import numpy as np

from tools.train_env_conf_validate import read_usr_conf

GRID = 128
DIRS = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
CARD = [0, 2, 4, 6]


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


def run_episode(env, uc, ei):
    env_obs = env.reset(usr_conf=uc)
    step = 0
    t0 = time.time()
    tick = 0
    while True:
        obs = env_obs["observation"]
        fs = obs["frame_state"]
        hpos = get_pos(fs["heroes"])
        monsters = fs.get("monsters", [])

        # print every monster's full dict
        out(f"s={step} hero={hpos} nmon={len(monsters)}")
        for m in monsters:
            out(f"    {m}")

        # explore: pick a passable cardinal cell
        view = parse_view(obs.get("map_info"))
        action = None
        order = [CARD[(tick + i) % 4] for i in range(4)]
        for d in order:
            dx, dz = DIRS[d]
            tr, tc = 10 + dx, 10 + dz
            if 0 <= tr < 21 and 0 <= tc < 21 and view[tr, tc] == 1:
                action = d
                break
        if action is None:
            action = 0

        env_reward, env_obs = env.step(action)
        if "observation" not in env_obs:
            out(f"step error at s={step}")
            return step
        step += 1
        tick += 1
        if env_obs.get("terminated") or env_obs.get("truncated"):
            out(f"EPISODE e{ei} END s={step} elapsed={time.time()-t0:.1f}s")
            return step
        if step >= 90:
            out(f"reached 90, stop diag")
            return step


def workflow(envs, agents, logger=None, monitor=None, *a, **k):
    out("PROBE v27 START")
    uc = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    env = envs[0]
    results = []
    for ei in range(2):
        try:
            results.append(run_episode(env, uc, ei))
        except Exception:
            import traceback
            out("CRASH:\n" + traceback.format_exc())
        time.sleep(1)
    out(f"RESULTS {results}")
    out("PROBE v27 END")
    time.sleep(3)
