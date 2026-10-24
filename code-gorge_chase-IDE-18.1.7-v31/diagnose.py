#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_rule_bot_mock import MockEnv
from agent_ppo.rule_bot import RuleBot


def diagnose(ratio=0.0, use_buff=False, seed=0):
    env = MockEnv(obstacle_ratio=ratio, use_buff=use_buff, seed=seed)
    bot = RuleBot()
    step = 0
    while True:
        fm = env.build_fm()
        legal = env.legal_actions()
        a = bot.act(fm, legal)
        term, trunc, fl = env.step(a)
        step += 1
        if term or trunc or step <= 12 or step % 10 == 0:
            d = min(math.hypot(m[0] - env.hero[0], m[1] - env.hero[1]) for m in env.monsters)
            print("step{} act{} hero{} nearest{:.1f} nMon{}".format(
                step, a, env.hero, d, len(env.monsters)))
        if term or trunc:
            print("RESULT", "WIN" if trunc else "FAIL", "at", step)
            break


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    use_buff = len(sys.argv) > 3
    diagnose(ratio=ratio, use_buff=use_buff, seed=seed)
