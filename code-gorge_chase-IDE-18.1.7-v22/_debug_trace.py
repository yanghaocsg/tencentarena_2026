#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Detailed trace of RuleBot in an open map to see how the monster closes in."""
import math
import test_rule_bot_mock_v2 as m2


def run(seed=4, lo=0, hi=999):
    env = m2.MockEnv(obstacle_ratio=0.0, use_buff=False, seed=seed, monster_ai="greedy")
    bot = m2.RuleBot()
    while True:
        fm = env.build_fm()
        legal = env.legal_actions()
        action = bot.act(fm, legal)
        term, trunc, _ = env.step(action)
        m = env.monsters[0]
        d = math.hypot(m[0] - env.hero[0], m[1] - env.hero[1])
        if lo <= env.step_no <= hi:
            ad = "MOVE" if action < 8 else "FLASH"
            print(
                f"step {env.step_no:3d} a={action}({ad}) hero {env.hero} mon {m} "
                f"dist {d:5.2f} hero_spd {env.hero_speed} mon_spd {env.monster_speed()}"
            )
        if term or trunc:
            print("DIED at", env.step_no)
            break


if __name__ == "__main__":
    # Print last 25 steps before death for seed 4 (~125 steps).
    run(seed=4, lo=100, hi=126)
