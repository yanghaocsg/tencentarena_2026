#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Check where the RuleBot dies (open map) and trace monster visibility."""
import math
import numpy as np
import test_rule_bot_mock_v2 as m2


def run_trace(seed=0):
    env = m2.MockEnv(obstacle_ratio=0.0, use_buff=False, seed=seed, monster_ai="greedy")
    bot = m2.RuleBot()
    last_vis = 0
    invis_runs = 0
    max_invis = 0
    while True:
        fm = env.build_fm()
        legal = env.legal_actions()
        # monster visible?
        vis = fm.all_monsters[0]["is_in_view"]
        if vis:
            last_vis = env.step_no
            invis_runs = 0
        else:
            invis_runs += 1
            max_invis = max(max_invis, invis_runs)
        action = bot.act(fm, legal)
        term, trunc, _ = env.step(action)
        if term or trunc:
            hx, hz = env.hero
            edge = min(hx, hz, 127 - hx, 127 - hz)
            print(
                f"seed{seed}: die step {env.step_no} hero {env.hero} edge_dist {edge} "
                f"last_saw_monster {last_vis} max_invisible_run {max_invis}"
            )
            return


if __name__ == "__main__":
    for s in range(12):
        run_trace(s)
