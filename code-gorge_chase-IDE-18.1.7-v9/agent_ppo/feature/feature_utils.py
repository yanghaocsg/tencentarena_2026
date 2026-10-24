#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Gorge Chase feature computation utility functions.
峡谷追猎特征计算工具函数。

Provides path planning, safety assessment, dead-end detection and other advanced feature calculations.
提供路径规划、安全评估、死角检测等高级特征计算功能。
"""

import numpy as np
from typing import Tuple, List, Optional
from collections import deque


# 8-directional move deltas: right, upper-right, up, upper-left, left, lower-left, down, lower-right
# 8方向移动增量：右, 右上, 上, 左上, 左, 左下, 下, 右下
DIRECTIONS_8 = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]


def is_valid_pos(x: int, z: int, map_size: int = 128) -> bool:
    """Check if position is within map bounds.

    检查位置是否在地图范围内。
    """
    return 0 <= x < map_size and 0 <= z < map_size


def is_passable(map_data: np.ndarray, x: int, z: int) -> bool:
    """Check if position is passable (1=passable, 0=obstacle).

    检查位置是否可通行（1表示可通行，0表示障碍物）。
    """
    if not is_valid_pos(x, z, map_data.shape[0]):
        return False
    return map_data[x, z] == 1


def _filter_visible_monsters(monsters: List[dict]) -> List[dict]:
    """Filter out visible monsters, excluding out-of-view ones with pos (-1,-1).

    过滤出有效（视野内）的怪物，排除 pos 为 (-1,-1) 的不可见怪物。
    """
    return [m for m in monsters if m.get("is_in_view", 0) == 1 and m["pos"]["x"] >= 0 and m["pos"]["z"] >= 0]


def calc_escape_routes(
    map_data: np.ndarray,
    hero_pos: Tuple[int, int],
    monsters: List[dict],
    max_depth: int = 10,
) -> np.ndarray:
    """Compute escape route safety scores for 8 directions.

    计算8个方向的逃跑路线安全评分。

    Args:
        map_data: map array, 1=passable, 0=obstacle / 地图数据，1可通行，0障碍
        hero_pos: hero position (x, z) / 小胖鹅当前位置
        monsters: monster list with pos field / 怪物列表
        max_depth: maximum search depth / 最大搜索深度

    Returns:
        np.array: 8D score array normalized to 0-1 / 8维归一化评分数组
    """
    scores = np.zeros(8)
    hx, hz = hero_pos
    monsters = _filter_visible_monsters(monsters)

    if monsters:
        monster_positions = [(m["pos"]["x"], m["pos"]["z"]) for m in monsters]
        avg_mx = np.mean([p[0] for p in monster_positions])
        avg_mz = np.mean([p[1] for p in monster_positions])
    else:
        avg_mx, avg_mz = hx, hz

    for dir_idx, (dx, dz) in enumerate(DIRECTIONS_8):
        move_dist = 0
        cx, cz = hx, hz
        for __ in range(max_depth):
            nx, nz = cx + dx, cz + dz
            if is_passable(map_data, nx, nz):
                move_dist += 1
                cx, cz = nx, nz
            else:
                break

        dist_score = move_dist / max_depth

        if monsters:
            dist_before = np.sqrt((hx - avg_mx) ** 2 + (hz - avg_mz) ** 2)
            test_x = hx + dx * min(3, move_dist)
            test_z = hz + dz * min(3, move_dist)
            dist_after = np.sqrt((test_x - avg_mx) ** 2 + (test_z - avg_mz) ** 2)
            escape_score = min(1.0, max(0.0, (dist_after - dist_before + 10) / 20))
        else:
            escape_score = 0.5

        scores[dir_idx] = 0.4 * dist_score + 0.6 * escape_score

    return scores


def detect_dead_ends(
    map_data: np.ndarray,
    hero_pos: Tuple[int, int],
    check_radius: int = 5,
) -> Tuple[int, float]:
    """Dead-end detection: check escape options around current position.

    死角检测：检测当前位置附近的逃跑选项。

    Returns:
        (escape_options, congestion_score)
    """
    hx, hz = hero_pos
    escape_options = 0
    blocked_count = 0
    total_checked = 0

    for dx, dz in DIRECTIONS_8:
        can_escape = True
        for step in range(1, 4):
            nx, nz = hx + dx * step, hz + dz * step
            if not is_passable(map_data, nx, nz):
                can_escape = False
                break
        if can_escape:
            escape_options += 1

    for dx in range(-check_radius, check_radius + 1):
        for dz in range(-check_radius, check_radius + 1):
            if dx == 0 and dz == 0:
                continue
            nx, nz = hx + dx, hz + dz
            total_checked += 1
            if not is_passable(map_data, nx, nz):
                blocked_count += 1

    congestion_score = blocked_count / max(total_checked, 1)
    return escape_options, congestion_score


def calc_treasure_priority(
    hero_pos: Tuple[int, int],
    treasures: List[dict],
    monsters: List[dict],
    map_data: np.ndarray,
) -> List[Tuple[int, float]]:
    """Compute treasure priority ranking.

    计算宝箱优先级排序。

    Returns:
        Priority list sorted descending: [(treasure_idx, priority_score), ...]
        按优先级降序排列: [(宝箱索引, 优先级分数), ...]
    """
    if not treasures:
        return []

    monsters = _filter_visible_monsters(monsters)
    hx, hz = hero_pos
    priorities = []

    monster_positions = []
    if monsters:
        for m in monsters:
            monster_positions.append((m["pos"]["x"], m["pos"]["z"]))

    for idx, treasure in enumerate(treasures):
        tx, tz = treasure["pos"]["x"], treasure["pos"]["z"]

        dist = np.sqrt((hx - tx) ** 2 + (hz - tz) ** 2)
        dist_score = max(0, 1 - dist / 180)

        safety_score = 1.0
        if monster_positions:
            min_monster_dist = float("inf")
            for mx, mz in monster_positions:
                m_dist = np.sqrt((tx - mx) ** 2 + (tz - mz) ** 2)
                min_monster_dist = min(min_monster_dist, m_dist)
            safety_score = min(1.0, min_monster_dist / 60)

        reachability = 1.0
        steps = max(1, int(dist / 10))
        blocked = 0
        for i in range(1, steps + 1):
            check_x = int(hx + (tx - hx) * i / steps)
            check_z = int(hz + (tz - hz) * i / steps)
            if not is_passable(map_data, check_x, check_z):
                blocked += 1
        reachability = 1 - blocked / max(steps, 1)

        priority = 0.4 * dist_score + 0.4 * safety_score + 0.2 * reachability
        priorities.append((idx, priority))

    priorities.sort(key=lambda x: x[1], reverse=True)
    return priorities


def find_nearest_safe_zone(
    map_data: np.ndarray,
    hero_pos: Tuple[int, int],
    monsters: List[dict],
    search_radius: int = 30,
    min_safe_dist: float = 50.0,
) -> Tuple[Optional[Tuple[int, int]], float]:
    """Find the nearest safe zone via BFS.

    通过BFS寻找最近的安全区域。

    Returns:
        (safe_pos, distance)
    """
    monsters = _filter_visible_monsters(monsters)
    if not monsters:
        return hero_pos, 0.0

    hx, hz = hero_pos
    monster_positions = [(m["pos"]["x"], m["pos"]["z"]) for m in monsters]

    visited = set()
    queue = deque([(hx, hz, 0)])
    visited.add((hx, hz))

    while queue:
        x, z, dist = queue.popleft()

        if dist > search_radius:
            break

        is_safe = True
        for mx, mz in monster_positions:
            if np.sqrt((x - mx) ** 2 + (z - mz) ** 2) < min_safe_dist:
                is_safe = False
                break

        if is_safe and is_passable(map_data, x, z):
            return (x, z), dist

        for dx, dz in DIRECTIONS_8:
            nx, nz = x + dx, z + dz
            if (nx, nz) not in visited and is_valid_pos(nx, nz, map_data.shape[0]):
                visited.add((nx, nz))
                if is_passable(map_data, nx, nz):
                    queue.append((nx, nz, dist + 1))

    return None, float("inf")


def calc_safety_heatmap(
    map_data: np.ndarray,
    hero_pos: Tuple[int, int],
    monsters: List[dict],
    grid_size: int = 4,
) -> np.ndarray:
    """Compute area safety heatmap.

    计算区域安全热力图。

    Returns:
        np.array: grid_size x grid_size safety score matrix (0-1) / 安全评分矩阵
    """
    map_size = map_data.shape[0]
    cell_size = map_size // grid_size
    heatmap = np.zeros((grid_size, grid_size))
    monsters = _filter_visible_monsters(monsters)

    monster_positions = []
    if monsters:
        for m in monsters:
            monster_positions.append((m["pos"]["x"], m["pos"]["z"]))

    for gi in range(grid_size):
        for gj in range(grid_size):
            center_x = gi * cell_size + cell_size // 2
            center_z = gj * cell_size + cell_size // 2

            if not monsters:
                heatmap[gi, gj] = 1.0
                continue

            min_dist = float("inf")
            for mx, mz in monster_positions:
                dist = np.sqrt((center_x - mx) ** 2 + (center_z - mz) ** 2)
                min_dist = min(min_dist, dist)

            safety = min(1.0, min_dist / 60.0)
            heatmap[gi, gj] = safety

    return heatmap


def check_treasure_safe_path(
    map_data: np.ndarray,
    hero_pos: Tuple[int, int],
    treasure_pos: Tuple[int, int],
    monsters: List[dict],
    danger_radius: float = 20.0,
) -> bool:
    """Check if the path to a treasure is safe from monsters.

    检查到宝箱的路径是否安全（无怪物威胁）。
    """
    monsters = _filter_visible_monsters(monsters)
    if not monsters:
        return True

    hx, hz = hero_pos
    tx, tz = treasure_pos
    monster_positions = [(m["pos"]["x"], m["pos"]["z"]) for m in monsters]

    dist = np.sqrt((tx - hx) ** 2 + (tz - hz) ** 2)
    steps = max(1, int(dist / 5))

    for i in range(steps + 1):
        px = hx + (tx - hx) * i / steps
        pz = hz + (tz - hz) * i / steps
        for mx, mz in monster_positions:
            if np.sqrt((px - mx) ** 2 + (pz - mz) ** 2) < danger_radius:
                return False

    return True
