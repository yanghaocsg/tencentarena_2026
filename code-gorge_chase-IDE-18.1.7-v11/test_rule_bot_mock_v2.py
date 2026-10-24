#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Realistic (vision-limited) mock test for the RuleBot.
规则 bot 的"真实视野"mock 测试。

Difference from test_rule_bot_mock.py:
- The original mock gave the bot the FULL map every step (god view, line 152
  fm.map_data[0] = self.grid), which is not how the real Preprocessor works.
- Here map_data[0] starts as all-passable and is updated ONLY inside the
  21x21 view each step (same as Preprocessor.update_map). Cells outside the
  view stay at the default 1 (assumed passable), so unseen walls are unknown.
区别：原 mock 每步把完整地图喂给 bot（上帝视角）；本测试让 map_data[0] 初始全
可走，每步只在 21x21 视野内更新（对齐真实 Preprocessor），视野外保持默认 1。

Run:  python test_rule_bot_mock_v2.py
"""

import math
import random
import sys
import os
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_ppo.rule_bot import RuleBot, MOVE_DIRS, FLASH_LEN  # noqa: E402

MAP_SIZE = 128
MAX_STEP = 1000
VIEW_RADIUS = 10
VIEW_SIZE = 21
MONSTER_INTERVAL = 300
MONSTER_SPEEDUP = 500
FLASH_CD = 100


def build_map(obstacle_ratio=0.0, seed=0):
    rng = random.Random(seed)
    grid = np.ones((MAP_SIZE, MAP_SIZE), dtype=int)
    if obstacle_ratio > 0:
        for _ in range(int(obstacle_ratio * 400)):
            bx, bz = rng.randint(5, MAP_SIZE - 12), rng.randint(5, MAP_SIZE - 12)
            w, h = rng.randint(2, 6), rng.randint(2, 6)
            grid[bx : bx + w, bz : bz + h] = 0
    return grid


def passable(grid, x, z):
    if x < 0 or z < 0 or x >= MAP_SIZE or z >= MAP_SIZE:
        return False
    return grid[x, z] == 1


def dist_bucket(d):
    edges = [30, 60, 90, 120, 150, 180]
    for i, e in enumerate(edges):
        if d < e:
            return i
    return 5


def rel_direction(hx, hz, mx, mz):
    dx, dz = mx - hx, mz - hz
    if dx == 0 and dz == 0:
        return 1
    angle = math.atan2(-dz, dx)
    deg = (math.degrees(angle) + 360) % 360
    mapping = [1, 2, 3, 4, 5, 6, 7, 8]
    idx = int(round(deg / 45.0)) % 8
    return mapping[idx]


def update_map(mem, view, pos):
    """Same update logic as Preprocessor.update_map (in-place)."""
    l = max(0, pos[0] - VIEW_RADIUS)
    r = min(MAP_SIZE, pos[0] + VIEW_RADIUS + 1)
    u = max(0, pos[1] - VIEW_RADIUS)
    d = min(MAP_SIZE, pos[1] + VIEW_RADIUS + 1)
    vx0 = l - (pos[0] - VIEW_RADIUS)
    vy0 = u - (pos[1] - VIEW_RADIUS)
    mem[l:r, u:d] = view[vx0:vx0 + (r - l), vy0:vy0 + (d - u)]


class MockFM:
    def __init__(self):
        self.cur_pos = (MAP_SIZE // 2, MAP_SIZE // 2)
        self.map_data = np.zeros((2, MAP_SIZE, MAP_SIZE), dtype=int)
        self.map_data[0] = np.ones((MAP_SIZE, MAP_SIZE), dtype=int)
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
        # Persistent obstacle MEMORY map: all-passable until seen.
        self.mem_obs = np.ones((MAP_SIZE, MAP_SIZE), dtype=int)
        self.hero_buff_remaining = 0
        self.flash_cd = 0
        self.monsters = []
        self.buff_pos = None
        self.buff_home = None
        self.buff_known = False
        self.buff_respawn_timer = 0
        self.buff_pickups = 0
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
            ang = self.rng.uniform(0, 2 * math.pi)
            mx = int(self.hero[0] + VIEW_RADIUS * math.cos(ang))
            mz = int(self.hero[1] + VIEW_RADIUS * math.sin(ang))
            mx = min(max(mx, 1), MAP_SIZE - 2)
            mz = min(mz, MAP_SIZE - 2)
            if not passable(self.grid, mx, mz):
                mx, mz = MAP_SIZE - 3, MAP_SIZE - 3
            self.monsters.append([mx, mz])
        else:
            self.monsters.append([self.hero[0], self.hero[1] - 12])

    def monster_speed(self):
        return 2 if self.step_no >= MONSTER_SPEEDUP else 1

    def _obstacle_view(self):
        """Build the 21x21 obstacle view from the TRUE grid at the hero."""
        view = np.ones((VIEW_SIZE, VIEW_SIZE), dtype=int)
        for i in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
            for j in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
                gx, gz = self.hero[0] + i, self.hero[1] + j
                if not passable(self.grid, gx, gz):
                    view[i + VIEW_RADIUS, j + VIEW_RADIUS] = 0
        return view

    def build_fm(self):
        # Update the persistent memory obstacle map only inside the view.
        update_map(self.mem_obs, self._obstacle_view(), self.hero)

        fm = MockFM()
        fm.cur_pos = (self.hero[0], self.hero[1])
        fm.map_data[0] = self.mem_obs  # vision-limited memory, NOT the full grid
        for i in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
            for j in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
                vx, vz = self.hero[0] + i, self.hero[1] + j
                if 0 <= vx < MAP_SIZE and 0 <= vz < MAP_SIZE:
                    self.visit[vx, vz] = 1
        fm.map_data[1] = self.visit
        fm.frame_no = self.step_no
        fm.skill_usable = self.flash_cd <= 0
        fm.speed_up = self.hero_buff_remaining > 0

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
        fm.treasure_list = []

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

        if self.buff_pos is not None and self.hero == self.buff_pos:
            self.hero_buff_remaining = 50
            self.hero_speed = 2
            self.buff_pos = None
            self.buff_pickups += 1
            self.buff_respawn_timer = 200

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

        if self.step_no == MONSTER_INTERVAL and len(self.monsters) < 2:
            self._spawn_monster(2)

        mspeed = self.monster_speed()
        if self.monster_ai == "bfs":
            dist = self._hero_dist_field()
            for _ in range(mspeed):
                for m in self.monsters:
                    self._move_monster_bfs(m, dist)
        else:
            for _ in range(mspeed):
                for m in self.monsters:
                    self._move_monster(m)

        for m in self.monsters:
            if math.hypot(m[0] - self.hero[0], m[1] - self.hero[1]) <= 1.5:
                return True, False, flashed
        if self.step_no >= MAX_STEP:
            return False, True, flashed
        return False, False, flashed

    def _move_hero(self, dx, dz):
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

    def _hero_dist_field(self):
        """BFS distance field from the hero (8-neighbour, walls block).
        Returns the shortest step count to reach the hero from every cell.
        """
        g = self.grid
        dist = np.full((MAP_SIZE, MAP_SIZE), 100000, dtype=np.int32)
        hx, hz = self.hero[0], self.hero[1]
        dist[hx, hz] = 0
        queue = [hx * MAP_SIZE + hz]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            x = cur // MAP_SIZE
            z = cur - x * MAP_SIZE
            nd = dist[x, z] + 1
            for dx, dz in MOVE_DIRS:
                nx, nz = x + dx, z + dz
                if 0 <= nx < MAP_SIZE and 0 <= nz < MAP_SIZE and g[nx, nz] and dist[nx, nz] > nd:
                    dist[nx, nz] = nd
                    queue.append(nx * MAP_SIZE + nz)
        return dist

    def _move_monster_bfs(self, m, dist):
        """Move to the neighbour with the smallest BFS distance to the hero;
        break ties by Euclidean distance so the monster cuts diagonals directly.
        """
        best, bestscore = None, 1e18
        hx, hz = self.hero[0], self.hero[1]
        for dx, dz in MOVE_DIRS:
            nx, nz = m[0] + dx, m[1] + dz
            if passable(self.grid, nx, nz):
                score = int(dist[nx, nz]) * 1000.0 + math.hypot(nx - hx, nz - hz)
                if score < bestscore:
                    bestscore, best = score, (nx, nz)
        if best is not None:
            m[0], m[1] = best


def run_once(obstacle_ratio, use_buff, seed, monster_ai="greedy"):
    env = MockEnv(obstacle_ratio=obstacle_ratio, use_buff=use_buff, seed=seed, monster_ai=monster_ai)
    bot = RuleBot()
    flashes = 0
    buffs = 0
    steps = 0
    while True:
        fm = env.build_fm()
        legal = env.legal_actions()
        action = bot.act(fm, legal)
        before = env.buff_pickups
        term, trunc, flashed = env.step(action)
        flashes += 1 if flashed else 0
        buffs += env.buff_pickups - before
        steps += 1
        if term or trunc:
            return steps, flashes, buffs, "WIN" if trunc else "FAIL"


def main():
    # (name, obstacle_ratio, use_buff, monster_ai, trials)
    # Fast greedy configs first; BFS configs are expensive (full-map BFS per step).
    configs = [
        ("open, no buff, greedy", 0.0, False, "greedy", 20),
        ("obstacles, no buff, greedy", 0.08, False, "greedy", 20),
        ("obstacles, buff, greedy", 0.08, True, "greedy", 20),
        ("open, no buff, BFS", 0.0, False, "bfs", 10),
        ("obstacles, buff, BFS", 0.08, True, "bfs", 10),
    ]
    for name, ratio, use_buff, ai, trials in configs:
        results, fl, bu, wins = [], 0, 0, 0
        for seed in range(trials):
            steps, flashes, buffs, result = run_once(ratio, use_buff, seed, monster_ai=ai)
            results.append(steps)
            fl += flashes
            bu += buffs
            if result == "WIN":
                wins += 1
        arr = np.array(results)
        print(
            f"[{name}] trials:{trials} WIN:{wins} mean:{arr.mean():.1f} "
            f"median:{np.median(arr):.0f} min:{arr.min()} max:{arr.max()} "
            f"flash:{fl/trials:.1f} buff:{bu/trials:.1f}"
        )


if __name__ == "__main__":
    main()
