#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Wall / corridor test for the RuleBot.
规则 bot 的长墙 / 走廊测试。

Verifies whether the bot can use walls to break pursuit: the monster must walk
around the wall while the hero gains distance.
验证 bot 能否利用长墙切断追击：怪物必须绕墙，英雄拉开距离。
"""

import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_rule_bot_mock import MockEnv, passable, MAP_SIZE  # noqa: E402
from agent_ppo.rule_bot import RuleBot  # noqa: E402


def build_wall_map():
    """Zigzag corridor maze: horizontal walls with alternating gaps.

    交错走廊迷宫：水平墙，缺口交替在左右。
    """
    grid = np.ones((MAP_SIZE, MAP_SIZE), dtype=int)
    walls = [20, 40, 60, 80, 100, 120]
    for i, wx in enumerate(walls):
        if i % 2 == 0:
            # Gap on the right side
            for z in range(2, 110):
                grid[wx, z] = 0
        else:
            # Gap on the left side
            for z in range(18, MAP_SIZE - 2):
                grid[wx, z] = 0
    return grid


def run(seed, max_steps=1000, verbose=False, use_buff=True, monster_ai="greedy"):
    env = MockEnv(obstacle_ratio=0.0, use_buff=use_buff, seed=seed, monster_ai=monster_ai)
    env.grid = build_wall_map()
    # Hero in the middle, monster below within view
    env.hero = [64, 64]
    env.monsters = [[64, 72]]
    if use_buff:
        # Place a buff reachable in the same corridor as the hero
        env.buff_pos = [70, 50]
    bot = RuleBot()
    step = 0
    flashes = 0
    while True:
        fm = env.build_fm()
        legal = env.legal_actions()
        a = bot.act(fm, legal)
        term, trunc, fl = env.step(a)
        step += 1
        flashes += 1 if fl else 0
        if verbose and (step % 50 == 0 or term or trunc):
            d = min(math.hypot(m[0] - env.hero[0], m[1] - env.hero[1]) for m in env.monsters)
            print("step{} act{} hero{} monsters{} nearest{:.1f}".format(
                step, a, env.hero, env.monsters, d))
        if term or trunc or step >= max_steps:
            if verbose:
                print("RESULT", "WIN" if (trunc or step >= max_steps) else "FAIL", "at", step, "flashes", flashes)
            return step, ("WIN" if (trunc or step >= max_steps) else "FAIL"), flashes


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ai = sys.argv[2] if len(sys.argv) > 2 else "greedy"
        run(int(sys.argv[1]), verbose=True, monster_ai=ai)
    else:
        for ai in ["greedy", "bfs"]:
            results = []
            wins = 0
            for seed in range(15):
                step, result, flashes = run(seed, monster_ai=ai)
                results.append(step)
                wins += 1 if result == "WIN" else 0
            arr = np.array(results)
            print("[{}] maze trials:15 WIN:{} mean_steps:{:.1f} median:{:.0f} min:{} max:{}".format(
                ai, wins, arr.mean(), np.median(arr), arr.min(), arr.max()))
