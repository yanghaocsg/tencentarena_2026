#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_rule_bot_mock import MockEnv
from test_walls import build_wall_map
from agent_ppo.rule_bot import RuleBot, MOVE_DIRS, MONSTER_RADIUS, HERO_RADIUS

env = MockEnv(obstacle_ratio=0.0, use_buff=True, seed=0)
env.grid = build_wall_map()
env.hero = [64, 58]
env.monsters = [[64, 66]]
env.buff_pos = [70, 50]
bot = RuleBot()

fm = env.build_fm()
legal = env.legal_actions()
x, z = fm.cur_pos
grid = fm.map_data[0]
monsters = bot._collect_monsters(fm)
treasures = bot._collect_treasures(fm)
buff = bot._collect_buff(fm)

hero_dist, hero_parent = bot._bfs_from(grid, x, z, HERO_RADIUS, speed=1)
monster_dist = bot._bfs_monsters(grid, monsters, MONSTER_RADIUS, 1)


def mdist(px, pz):
    d = monster_dist.get((px, pz), None)
    return MONSTER_RADIUS + 4.0 if d is None else d


dir_safe = bot._dir_safe_depths(grid, x, z, mdist)

buff_dist = None
buff_safe = False
if buff is not None:
    buff_dist, _ = bot._bfs_from(grid, buff[0], buff[1], HERO_RADIUS, speed=1)
    h_to_buff = buff_dist.get((x, z), None)
    if h_to_buff is not None and mdist(buff[0], buff[1]) > h_to_buff + 1.0:
        buff_safe = True

print("hero", (x, z), "monsters", monsters, "buff", buff, "buff_safe", buff_safe)
print("h_to_buff", buff_dist.get((x, z)) if buff_dist else None,
      "mdist(buff)", mdist(buff[0], buff[1]) if buff else None)
print("dir_safe", dir_safe)

cur_bdist = buff_dist.get((x, z), None) if (buff_dist is not None and buff_safe) else None

for d in range(8):
    if legal[d] != 1:
        continue
    dx, dz = MOVE_DIRS[d]
    cell = (x + dx, z + dz)
    if not bot._passable(grid, cell[0], cell[1]):
        print("dir{} BLOCKED".format(d))
        continue
    hd = hero_dist.get(cell)
    md = mdist(cell[0], cell[1])
    margin = md - hd
    s = margin * 1.4
    s += min(md, 12.0) * 0.35
    first_d = bot._first_step_dir(x, z, cell, hero_parent)
    ds = 0.0
    if first_d is not None:
        ds = (dir_safe[first_d] / 30.0) * 2.4
        s += ds
    bg = 0.0
    if cur_bdist is not None and margin >= 1.0:
        cell_bdist = buff_dist.get(cell, None)
        if cell_bdist is not None and cell_bdist < cur_bdist:
            bg = (cur_bdist - cell_bdist) * 0.8
            s += bg
    print("dir{} cell{} hd{} md{:.1f} margin{:.1f} dirSafe{:.2f} buff{:.2f} SCORE{:.2f}".format(
        d, cell, hd, md, margin, ds, bg, s))
