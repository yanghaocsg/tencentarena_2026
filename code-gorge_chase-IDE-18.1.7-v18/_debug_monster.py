#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Debug: compare greedy vs BFS monster movement using a straight-line flee."""
import math
import numpy as np
import test_rule_bot_mock_v2 as m2


def straight_run(ai, seed=0):
    """Hero always runs straight away from monster (purely in -x,-z direction)."""
    env = m2.MockEnv(obstacle_ratio=0.0, use_buff=False, seed=seed, monster_ai=ai)
    # Force hero flee direction: move down-right? choose opposite of monster
    m = env.monsters[0]
    # flee direction opposite to monster
    dx = -1 if m[0] > env.hero[0] else 1
    dz = -1 if m[1] > env.hero[1] else 1
    # pick the move action whose dir matches
    flee_action = None
    for d, (ax, az) in enumerate(m2.MOVE_DIRS):
        if ax == dx and az == dz:
            flee_action = d
    while True:
        env.build_fm()
        legal = env.legal_actions()
        if legal[flee_action] == 1:
            a = flee_action
        else:
            a = 0
        term, trunc, _ = env.step(a)
        if term or trunc:
            return env.step_no, "WIN" if trunc else "FAIL"


def trace_run(ai, seed=0, n=15):
    env = m2.MockEnv(obstacle_ratio=0.0, use_buff=False, seed=seed, monster_ai=ai)
    m = env.monsters[0]
    dx = -1 if m[0] > env.hero[0] else 1
    dz = -1 if m[1] > env.hero[1] else 1
    flee_action = None
    for d, (ax, az) in enumerate(m2.MOVE_DIRS):
        if ax == dx and az == dz:
            flee_action = d
    print(f"--- {ai} trace (hero starts {env.hero}, monster {m}, flee {flee_action}) ---")
    for i in range(n):
        env.build_fm()
        legal = env.legal_actions()
        a = flee_action if legal[flee_action] else 0
        term, trunc, _ = env.step(a)
        mm = env.monsters[0]
        d = math.hypot(mm[0] - env.hero[0], mm[1] - env.hero[1])
        print(f"step {env.step_no}: hero {env.hero} mon {mm} dist {d:.2f} speed {env.monster_speed()}")
        if term or trunc:
            print("ended", env.step_no)
            break


if __name__ == "__main__":
    for ai in ["greedy", "bfs"]:
        trace_run(ai, seed=0, n=12)
    for ai in ["greedy", "bfs"]:
        ds = [straight_run(ai, seed=s)[0] for s in range(8)]
        print(f"straight flee {ai}: {ds} mean {np.mean(ds):.0f}")
