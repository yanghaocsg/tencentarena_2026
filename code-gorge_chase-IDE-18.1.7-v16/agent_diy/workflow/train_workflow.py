#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""v16: print-based fixed-action diagnostic.

v15 showed the fixed action (act=5) also stalls at s=2, but the VIEW s=2
line and the STEP TIMEOUT (logger.error) never appeared. v13d proved that
plain print(flush=True) is reliable while logger.info/error can fail to
flush in the threaded/ending phase. So v16:

- Use print(..., flush=True) for EVERYTHING (no logger).
- Fixed action act=5 every step.
- Each step print: obs keys, map_info type/len, view shape/unique,
  center row/col, hero/monster/target.
- env.step runs in a worker thread with a 3s timeout; on timeout print
  a clear marker, then reset and try a fresh episode (up to 3 episodes).
"""
import sys
import threading
import time

import numpy as np

from tools.train_env_conf_validate import read_usr_conf

GRID = 128
PASSABLE, BLOCKED, UNKNOWN = 1, 0, -1
DIRS = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
FIXED_ACT = 5
STEP_TIMEOUT = 3.0
MAX_STEPS = 30
N_EPISODES = 3


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


def describe_map_info(mi):
    out(f"  map_info type={type(mi).__name__} len={len(mi) if mi is not None else 'None'}")
    if not mi:
        return None
    out(f"  elem0 type={type(mi[0]).__name__}")
    if isinstance(mi[0], dict):
        out(f"  elem0 keys={list(mi[0].keys())}")
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
        return None, t
    return box[0], t


def probe(env, usr_conf, ep, logger):
    out(f"--- v16 episode {ep} reset ---")
    env_obs = env.reset(usr_conf)
    for step in range(MAX_STEPS):
        out(f"[ep{ep}] ===== s={step} =====")
        out(f"  obs keys={list(env_obs.keys())}")
        obs = env_obs["observation"]
        out(f"  observation keys={list(obs.keys())}")
        fs = obs["frame_state"]
        hero = fs["heroes"]
        hpos = get_pos(hero)
        monsters = [get_pos(m) for m in fs.get("monsters", [])]
        monsters = [m for m in monsters if m]
        organs = fs.get("organs", [])
        org_info = [(o.get("sub_type"), get_pos(o), o.get("status")) for o in organs]
        out(f"  hero={hpos} mon={monsters} org={org_info}")

        view = describe_map_info(obs.get("map_info"))
        act = FIXED_ACT
        tgt = (hpos[0] + DIRS[act][0], hpos[1] + DIRS[act][1])
        out(f"  act={act} tgt={tgt}")
        if view is not None:
            uniq = sorted(set(int(v) for v in view.flat))
            out(f"  view shape={view.shape} uniq={uniq}")
            out(f"  centerROW={view[10].tolist()}")
            out(f"  centerCOL={view[:, 10].tolist()}")

        r, t = step_with_timeout(env, act, STEP_TIMEOUT)
        if r is None:
            out(f"  !!! STEP TIMEOUT s={step} act={act} hero={hpos} tgt={tgt}")
            return "timeout"
        if isinstance(r, tuple) and len(r) == 2 and r[0] == "EXC":
            out(f"  !!! STEP EXC s={step}: {repr(r[1])}")
            return "exc"

        env_reward, env_obs = r
        if env_obs.get("terminated") or env_obs.get("truncated"):
            out(f"  END s={step} term={env_obs.get('terminated')} "
                f"trunc={env_obs.get('truncated')}")
            return "end"
    out(f"  reached MAX_STEPS={MAX_STEPS}")
    return "max"


def workflow(envs, agents, logger=None, monitor=None, *a, **k):
    out("PROBE v16 START")
    uc = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    env = envs[0]
    for ep in range(N_EPISODES):
        try:
            res = probe(env, uc, ep, logger)
            out(f"--- episode {ep} result={res} ---")
        except Exception as e:  # noqa
            import traceback
            out("CRASH:\n" + traceback.format_exc())
        time.sleep(1)
    out("PROBE v16 END")
    time.sleep(3)
