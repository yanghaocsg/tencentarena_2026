#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Local mock-environment test for the RuleBot (no kaiwudrl / simulator needed).
规则 bot 的本地 mock 环境测试（无需 kaiwudrl / 模拟器）。

It simulates a simplified Gorge Chase: greedy monsters, a 128x128 map (open or
with scattered obstacles), flash skill, and buff. It reports how long the bot
survives, how many times it flashes, and whether it reaches 1000 steps (WIN).
模拟简化的峡谷追猎：贪心怪物、128×128 地图（开阔或散布障碍）、穿梭技能、buff，
统计 bot 存活步数、穿梭次数、是否活满 1000 步（WIN）。

Run:  python test_rule_bot_mock.py
"""

import math
import random
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_ppo.rule_bot import RuleBot, MOVE_DIRS, FLASH_LEN  # noqa: E402

MAP_SIZE = 128
MAX_STEP = 1000
VIEW_RADIUS = 10
MONSTER_INTERVAL = 300
MONSTER_SPEEDUP = 500
FLASH_CD = 100


def build_map(obstacle_ratio=0.0, seed=0):
    """Build a passable map. obstacle_ratio adds scattered obstacle blocks.

    生成可走地图，obstacle_ratio 加入散布障碍块。
    """
    rng = random.Random(seed)
    grid = np.ones((MAP_SIZE, MAP_SIZE), dtype=int)
    # Keep a clear spawn region around the center
    # 中心出生区保持开阔
    if obstacle_ratio > 0:
        for _ in range(int(obstacle_ratio * 400)):
            bx, bz = rng.randint(5, MAP_SIZE - 12), rng.randint(5, MAP_SIZE - 12)
            w, h = rng.randint(2, 6), rng.randint(2, 6)
            grid[bx : bx + w, bz : bz + h] = 0
    # Border ring is open by default here
    return grid


def passable(grid, x, z):
    if x < 0 or z < 0 or x >= MAP_SIZE or z >= MAP_SIZE:
        return False
    return grid[x, z] == 1


def dist_bucket(d):
    # hero_l2_distance buckets in [0,5]
    # 与环境一致的距离桶 0-5
    edges = [30, 60, 90, 120, 150, 180]
    for i, e in enumerate(edges):
        if d < e:
            return i
    return 5


def rel_direction(hx, hz, mx, mz):
    # Monster-relative direction 1-8, aligned with the preprocessor convention
    # 怪物相对方向 1-8，对齐 preprocessor 约定
    dx, dz = mx - hx, mz - hz
    if dx == 0 and dz == 0:
        return 1
    angle = math.atan2(-dz, dx)  # preprocessor uses z-up for direction encoding
    # direction vectors 1:(1,0) 2:(1,-1) 3:(0,-1) 4:(-1,-1) ...
    deg = (math.degrees(angle) + 360) % 360
    mapping = [1, 2, 3, 4, 5, 6, 7, 8]
    idx = int(round(deg / 45.0)) % 8
    return mapping[idx]


class MockFM:
    """Mock Preprocessor exposing only the fields RuleBot uses."""

    def __init__(self):
        self.cur_pos = (MAP_SIZE // 2, MAP_SIZE // 2)
        self.map_data = np.zeros((2, MAP_SIZE, MAP_SIZE), dtype=int)
        self.all_monsters = []
        self.treasure_list = []
        self.buff_pos = self.cur_pos
        self.buff_found = False
        self.buff_status = False
        self.buff_cooldown = 0
        self.skill_usable = True
        self.speed_up = False
        self.frame_no = 0


class MockEnv:
    def __init__(self, obstacle_ratio=0.0, use_buff=True, seed=0, monster_ai="greedy"):
        self.grid = build_map(obstacle_ratio, seed)
        self.rng = random.Random(seed + 1)
        self.use_buff = use_buff
        self.monster_ai = monster_ai
        self.hero = [MAP_SIZE // 2, MAP_SIZE // 2]
        self.hero_speed = 1
        self.visit = np.zeros((MAP_SIZE, MAP_SIZE), dtype=int)
        self.hero_buff_remaining = 0
        self.flash_cd = 0
        self.monsters = []  # [x,z, exists]
        self.buff_pos = None
        self.buff_home = None
        self.buff_known = False
        self.buff_respawn_timer = 0
        self.buff_pickups = 0
        self.treasures = []
        self.step_no = 0
        self._place_buff()
        self._spawn_monster(1)

    def _place_buff(self):
        if self.use_buff:
            for _ in range(50):
                bx, bz = self.rng.randint(15, MAP_SIZE - 15), self.rng.randint(15, MAP_SIZE - 15)
                if passable(self.grid, bx, bz):
                    self.buff_pos = [bx, bz]
                    self.buff_home = [bx, bz]
                    return

    def _spawn_monster(self, idx):
        if idx == 1:
            # Spawn at the edge of the hero view region
            # 在英雄视野边缘出生
            ang = self.rng.uniform(0, 2 * math.pi)
            mx = int(self.hero[0] + VIEW_RADIUS * math.cos(ang))
            mz = int(self.hero[1] + VIEW_RADIUS * math.sin(ang))
            mx = min(max(mx, 1), MAP_SIZE - 2)
            mz = min(max(mz, 1), MAP_SIZE - 2)
            if not passable(self.grid, mx, mz):
                mx, mz = MAP_SIZE - 3, MAP_SIZE - 3
            self.monsters.append([mx, mz])
        else:
            # Monster 2 spawns near the hero (hero's position 10 steps ago approximated)
            self.monsters.append([self.hero[0], self.hero[1] - 12])

    def monster_speed(self):
        return 2 if self.step_no >= MONSTER_SPEEDUP else 1

    def build_fm(self):
        fm = MockFM()
        fm.cur_pos = (self.hero[0], self.hero[1])
        fm.map_data[0] = self.grid
        # Mark cells within view radius as visited, expose the visit mask
        # 标记英雄视野内格子为已访问，并暴露 visit 掩码
        for i in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
            for j in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
                vx, vz = self.hero[0] + i, self.hero[1] + j
                if 0 <= vx < MAP_SIZE and 0 <= vz < MAP_SIZE:
                    self.visit[vx, vz] = 1
        fm.map_data[1] = self.visit
        fm.frame_no = self.step_no
        fm.skill_usable = self.flash_cd <= 0
        fm.speed_up = self.hero_buff_remaining > 0

        # Monsters, visibility and fields
        all_m = []
        for m in self.monsters:
            d = math.hypot(m[0] - self.hero[0], m[1] - self.hero[1])
            visible = d <= VIEW_RADIUS
            all_m.append(
                {
                    "pos": {"x": m[0] if visible else -1, "z": m[1] if visible else -1},
                    "is_in_view": 1 if visible else 0,
                    "hero_l2_distance": dist_bucket(d),
                    "hero_relative_direction": rel_direction(self.hero[0], self.hero[1], m[0], m[1]),
                }
            )
        fm.all_monsters = all_m

        # Treasures (none in this survival-focused mock)
        fm.treasure_list = []

        # Buff: expose the fixed location, availability status and cooldown.
        # buff：暴露固定位置、可用状态与冷却时间。
        if self.buff_pos is not None:
            bd = math.hypot(self.buff_pos[0] - self.hero[0], self.buff_pos[1] - self.hero[1])
            if bd <= VIEW_RADIUS:
                self.buff_known = True
        if self.buff_known and self.buff_home is not None:
            fm.buff_found = True
            fm.buff_pos = (self.buff_home[0], self.buff_home[1])
            if self.buff_pos is not None:
                fm.buff_status = True
            else:
                fm.buff_status = False
                fm.buff_cooldown = self.buff_respawn_timer
        return fm

    def legal_actions(self):
        legal = [0] * 16
        for d, (dx, dz) in enumerate(MOVE_DIRS):
            if passable(self.grid, self.hero[0] + dx, self.hero[1] + dz):
                legal[d] = 1
        if self.flash_cd <= 0:
            for d in range(8):
                legal[8 + d] = 1
        return legal

    def step(self, action):
        self.step_no += 1
        flashed = False
        if action < 8:
            dx, dz = MOVE_DIRS[action]
            self._move_hero(dx * self.hero_speed, dz * self.hero_speed)
        else:
            d = action - 8
            dx, dz = MOVE_DIRS[d]
            length = FLASH_LEN[d]
            self._move_hero(dx * length, dz * length)
            self.flash_cd = FLASH_CD
            flashed = True

        # Pick up buff
        if self.buff_pos is not None and self.hero == self.buff_pos:
            self.hero_buff_remaining = 50
            self.hero_speed = 2
            self.buff_pos = None
            self.buff_pickups += 1
            # Respawn at the SAME fixed location after cooldown (default 200)
            # 冷却（默认 200 步）后在原位置刷新
            self.buff_respawn_timer = 200

        # Buff respawn countdown (same location)
        if self.buff_pos is None and self.buff_respawn_timer > 0:
            self.buff_respawn_timer -= 1
            if self.buff_respawn_timer == 0 and self.buff_home is not None:
                self.buff_pos = [self.buff_home[0], self.buff_home[1]]

        if self.hero_buff_remaining > 0:
            self.hero_buff_remaining -= 1
            if self.hero_buff_remaining == 0:
                self.hero_speed = 1

        if self.flash_cd > 0:
            self.flash_cd -= 1

        # Spawn monster 2
        if self.step_no == MONSTER_INTERVAL and len(self.monsters) < 2:
            self._spawn_monster(2)

        # Move monsters toward the hero
        mspeed = self.monster_speed()
        if self.monster_ai == "bfs":
            # BFS from the hero: a shared "next-step toward hero" field
            # 从英雄 BFS：所有怪物共用"朝英雄的下一步"场
            parent = self._hero_parent_field()
            for _ in range(mspeed):
                for m in self.monsters:
                    p = parent[m[0], m[1]]
                    if p >= 0:
                        px, pz = p // MAP_SIZE, p % MAP_SIZE
                        m[0], m[1] = px, pz
        else:
            for _ in range(mspeed):
                for m in self.monsters:
                    self._move_monster(m)

        # Collision check (hero within 3x3 of a monster, i.e. distance <= 1)
        for m in self.monsters:
            if math.hypot(m[0] - self.hero[0], m[1] - self.hero[1]) <= 1.5:
                return True, False, flashed
        if self.step_no >= MAX_STEP:
            return False, True, flashed
        return False, False, flashed

    def _move_hero(self, dx, dz):
        # Move along the vector, stopping at last passable cell on obstacle
        # 沿向量移动，遇障碍停在最后可走格
        sx = 1 if dx > 0 else (-1 if dx < 0 else 0)
        sz = 1 if dz > 0 else (-1 if dz < 0 else 0)
        steps = max(abs(dx), abs(dz))
        for _ in range(steps):
            nx, nz = self.hero[0] + sx, self.hero[1] + sz
            if passable(self.grid, nx, nz):
                self.hero = [nx, nz]

    def _move_monster(self, m):
        best, bestd = None, 1e18
        for dx, dz in MOVE_DIRS:
            nx, nz = m[0] + dx, m[1] + dz
            if passable(self.grid, nx, nz):
                d = math.hypot(nx - self.hero[0], nz - self.hero[1])
                if d < bestd:
                    bestd, best = d, (nx, nz)
        if best is not None:
            m[0], m[1] = best

    def _hero_parent_field(self):
        """BFS from the hero over the whole map.

        Returns a parent array where parent[x,z] is the encoded cell (x*MAP_SIZE+z)
        of the next cell toward the hero, or -1 if unreachable.
        从英雄对全图 BFS，parent[x,z] 为朝英雄的下一步（编码 x*MAP_SIZE+z），不可达为 -1。
        """
        g = self.grid
        parent = np.full((MAP_SIZE, MAP_SIZE), -1, dtype=np.int32)
        hx, hz = self.hero[0], self.hero[1]
        root = hx * MAP_SIZE + hz
        parent[hx, hz] = root
        queue = [root]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            x = cur // MAP_SIZE
            z = cur - x * MAP_SIZE
            for dx, dz in MOVE_DIRS:
                nx, nz = x + dx, z + dz
                if 0 <= nx < MAP_SIZE and 0 <= nz < MAP_SIZE and g[nx, nz] and parent[nx, nz] < 0:
                    parent[nx, nz] = cur
                    queue.append(nx * MAP_SIZE + nz)
        return parent


def run_once(obstacle_ratio, use_buff, seed, monster_ai="greedy", verbose=False):
    env = MockEnv(obstacle_ratio=obstacle_ratio, use_buff=use_buff, seed=seed, monster_ai=monster_ai)
    bot = RuleBot()
    flashes = 0
    steps = 0
    while True:
        fm = env.build_fm()
        legal = env.legal_actions()
        action = bot.act(fm, legal)
        term, trunc, flashed = env.step(action)
        flashes += 1 if flashed else 0
        steps += 1
        if term or trunc:
            return steps, flashes, "WIN" if trunc else "FAIL"


def main():
    configs = [
        ("open map, no buff, greedy", 0.0, False, "greedy"),
        ("open map, no buff, BFS", 0.0, False, "bfs"),
        ("obstacles, with buff, BFS", 0.08, True, "bfs"),
    ]
    trials = 20
    for name, ratio, use_buff, ai in configs:
        results = []
        wins = 0
        flash_total = 0
        for seed in range(trials):
            steps, flashes, result = run_once(ratio, use_buff, seed, monster_ai=ai)
            results.append(steps)
            flash_total += flashes
            if result == "WIN":
                wins += 1
        arr = np.array(results)
        print(
            f"[{name}] trials:{trials} WIN:{wins} mean_steps:{arr.mean():.1f} "
            f"median:{np.median(arr):.0f} min:{arr.min()} max:{arr.max()} "
            f"mean_flashes:{flash_total/trials:.1f}"
        )


if __name__ == "__main__":
    main()
