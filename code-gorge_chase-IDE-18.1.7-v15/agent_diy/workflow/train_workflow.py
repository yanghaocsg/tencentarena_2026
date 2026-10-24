#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""v15: fixed-action diagnostic.

Goal: isolate why v14's env.step hung at s=2.
- Do NOT use BFS. Always send a FIXED action (act=5, lower-left, which
  moves away from the monster at spawn).
- Wrap env.step with a hard timeout (3s, worker thread).
- Print every step: hero/monster/organ positions, target cell, and for
  the first steps the raw view (unique values + center row/col) so the
  map_info axis orientation can be read off.

If fixed action runs fine -> the hang is caused by BFS's chosen action.
If fixed action also hangs at the same target cell -> it is an env-side
issue tied to that cell.
"""
import threading
import time

import numpy as np

from tools.train_env_conf_validate import read_usr_conf

GRID = 128
PASSABLE, BLOCKED, UNKNOWN = 1, 0, -1
DIRS = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]

# Fixed action used every step (lower-left, away from the spawn monster).
FIXED_ACT = 5
STEP_TIMEOUT = 3.0
MAX_STEPS = 20


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


def step_with_timeout(env, act, timeout):
    box = []

    def worker():
        try:
            box.append(env.step(act))
        except Exception as e:  # noqa
            box.append(("EXC", e))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None
    return box[0]


def probe(envs, usr_conf, ei, logger):
    env = envs[ei]
    env_obs = env.reset(usr_conf)
    for step in range(MAX_STEPS):
        obs = env_obs["observation"]
        fs = obs["frame_state"]
        # frame_state MUST be touched before map_info.
        hero = fs["heroes"]
        hpos = get_pos(hero)
        monsters = [get_pos(m) for m in fs.get("monsters", [])]
        monsters = [m for m in monsters if m]
        organs = fs.get("organs", [])
        org_info = [(o.get("sub_type"), get_pos(o), o.get("status")) for o in organs]
        view = parse_view(obs.get("map_info"))

        act = FIXED_ACT
        tgt = (hpos[0] + DIRS[act][0], hpos[1] + DIRS[act][1])
        logger.info(
            f"v15 e{ei} s={step} hero={hpos} mon={monsters} act={act} "
            f"tgt={tgt} org={org_info}")

        if step < 4 and view is not None:
            uniq = sorted(set(int(v) for v in view.flat))
            logger.info(
                f"VIEW e{ei} s={step} uniq={uniq} "
                f"centerROW={view[10].tolist()} "
                f"centerCOL={view[:, 10].tolist()}")

        r = step_with_timeout(env, act, STEP_TIMEOUT)
        if r is None:
            logger.error(
                f"v15 e{ei} STEP TIMEOUT s={step} act={act} "
                f"hero={hpos} tgt={tgt} mon={monsters}")
            return
        if isinstance(r, tuple) and len(r) == 2 and r[0] == "EXC":
            logger.error(f"v15 e{ei} STEP EXC s={step}: {repr(r[1])}")
            return

        env_reward, env_obs = r
        if env_obs.get("terminated") or env_obs.get("truncated"):
            logger.info(
                f"v15 e{ei} END s={step} term={env_obs.get('terminated')} "
                f"trunc={env_obs.get('truncated')}")
            return
    logger.info(f"v15 e{ei} reached MAX_STEPS={MAX_STEPS}")


def workflow(envs, agents, logger=None, monitor=None, *a, **k):
    logger.info("PROBE v15 START")
    uc = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    for ei in range(len(envs)):
        try:
            probe(envs, uc, ei, logger)
        except Exception as e:  # noqa
            import traceback
            logger.error(traceback.format_exc())
    logger.info("PROBE v15 END")
    time.sleep(3)
