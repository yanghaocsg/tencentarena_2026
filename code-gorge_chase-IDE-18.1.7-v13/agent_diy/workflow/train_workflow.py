#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Diagnostic v13d: touch frame_state first, use print(flush), per-cell access."""
import sys
import time
import numpy as np
from tools.train_env_conf_validate import read_usr_conf


def probe_episode(envs, usr_conf, logger):
    env = envs[0]
    env_obs = env.reset(usr_conf)

    def p(*a):
        print("DIAG", *a, flush=True)
        sys.stdout.flush()

    obs = env_obs["observation"]
    fs = obs["frame_state"]
    # Touch frame_state fields BEFORE iterating map_info (as v12 does).
    hero = fs["heroes"]
    monsters = fs.get("monsters", [])
    organs = fs.get("organs", [])
    p("touched frame_state: hero=", hero.get("pos"),
      "n_mon=", len(monsters), "n_org=", len(organs))

    mi = obs.get("map_info")
    p("mi type=", type(mi).__name__, "len=", len(mi))

    for ri in range(len(mi)):
        p("row", ri, "getting mi[ri]")
        row = mi[ri]
        p("row", ri, "type=", type(row).__name__, "len=", len(row))
        vals = []
        for ci in range(len(row)):
            x = row[ci]
            vals.append(int(x))
        p("row", ri, "vals=", vals)

    p("ALL DONE")
    time.sleep(5)


def workflow(envs, agents, logger=None, monitor=None, *a, **k):
    print("DIAG START", flush=True)
    uc = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    try:
        probe_episode(envs, uc, logger)
    except Exception as e:
        print("DIAG crash:", repr(e), flush=True)
        import traceback
        print(traceback.format_exc(), flush=True)
    print("DIAG END", flush=True)
    time.sleep(5)
