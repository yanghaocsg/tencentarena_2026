#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_rule_bot_mock import MockEnv
from agent_ppo.rule_bot import RuleBot, MOVE_DIRS, SAFE_TIME, THREAT_RADIUS

env = MockEnv(obstacle_ratio=0.0, use_buff=False, seed=0)
bot = RuleBot()

fm = env.build_fm()
legal = env.legal_actions()
x, z = fm.cur_pos
grid = fm.map_data[0]
monsters = bot._collect_monsters(fm)
field = bot._threat_field(grid, monsters)
print("hero", (x, z), "monsters", monsters)
print("cur BFS", field.get((x, z)), "arrival", field.get((x, z)) / 1.0, "SAFE_TIME", SAFE_TIME)
away, tangent = bot._orbit_vectors(x, z, monsters)
print("away", away, "tangent", tangent)

for d in range(8):
    dx, dz = MOVE_DIRS[d]
    lx, lz = x + dx, z + dz
    bfs = field.get((lx, lz), None)
    arr = (THREAT_RADIUS + 2.0) if bfs is None else bfs
    dot_away = dx * away[0] + dz * away[1]
    dot_side = dx * tangent[0] + dz * tangent[1]
    print(
        "dir{} move{} BFS{} arr{:.1f} dotAway{:.2f} dotSide{:.2f} open{:.2f}".format(
            d, (dx, dz), bfs, arr, dot_away, dot_side, bot._openness(lx, lz, grid)
        )
    )

print("BOT CHOOSES", bot.act(fm, legal))
