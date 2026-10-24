#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Gorge Chase.
峡谷追猎特征预处理器。

Generates 480D feature vector + 20D auxiliary task labels + phased reward.
生成480维特征向量 + 20维辅助任务标签 + 阶段性奖励。
"""


import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from agent_ppo.feature.feature_utils import (
    calc_escape_routes,
    detect_dead_ends,
    find_nearest_safe_zone,
    check_treasure_safe_path,
    calc_treasure_priority,
    calc_safety_heatmap,
)


def norm_val(v, max_v, min_v=0):
    """Normalize value to [0, 1] range.

    将值归一化到 [0, 1] 范围。
    """
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)


@dataclass
class TemporalState:
    """Temporal state buffer for tracking monster historical trajectories.

    时序状态缓冲区，用于追踪怪物历史轨迹。
    """

    history_len: int = 5
    monster_history: deque = field(default_factory=lambda: deque(maxlen=5))
    hero_position_history: deque = field(default_factory=lambda: deque(maxlen=5))
    action_history: deque = field(default_factory=lambda: deque(maxlen=5))

    def __post_init__(self):
        self.monster_history = deque(maxlen=self.history_len)
        self.hero_position_history = deque(maxlen=self.history_len)
        self.action_history = deque(maxlen=self.history_len)

    def reset(self):
        self.monster_history.clear()
        self.hero_position_history.clear()
        self.action_history.clear()

    def update(self, monsters: List[dict], hero_pos: Tuple[int, int], action: int):
        monster_snapshot = {}
        for m in monsters:
            mid = m.get("monster_id", 0)
            pos = m.get("pos", {})
            x, z = pos.get("x", 0), pos.get("z", 0)

            vx, vz = 0, 0
            if len(self.monster_history) > 0:
                last_snapshot = self.monster_history[-1]
                if mid in last_snapshot:
                    last_x, last_z, __, __ = last_snapshot[mid]
                    vx = x - last_x
                    vz = z - last_z

            monster_snapshot[mid] = (x, z, vx, vz)

        self.monster_history.append(monster_snapshot)
        self.hero_position_history.append(hero_pos)
        self.action_history.append(action)

    def get_monster_trajectory(self, monster_id: int) -> List[Tuple[int, int, int, int]]:
        trajectory = []
        for snapshot in self.monster_history:
            if monster_id in snapshot:
                trajectory.append(snapshot[monster_id])
            else:
                trajectory.append((0, 0, 0, 0))
        return trajectory

    def get_velocity_trend(self, monster_id: int) -> Tuple[float, float]:
        trajectory = self.get_monster_trajectory(monster_id)
        if len(trajectory) < 2:
            return (0.0, 0.0)

        velocities = [(t[2], t[3]) for t in trajectory if t != (0, 0, 0, 0)]
        if len(velocities) < 2:
            return (0.0, 0.0)

        recent_vx, recent_vz = velocities[-1]
        if len(velocities) >= 3:
            prev_vx = np.mean([v[0] for v in velocities[:-1]])
            prev_vz = np.mean([v[1] for v in velocities[:-1]])
        else:
            prev_vx, prev_vz = velocities[-2]

        ax = recent_vx - prev_vx
        az = recent_vz - prev_vz
        return (ax, az)


class Preprocessor:
    """Feature preprocessor for Gorge Chase.

    峡谷追猎特征预处理器。
    """

    def __init__(self):
        self.move_action_num = 8
        self.skill_action_num = 8
        self.grid_map_size = 128
        self.visit_view_mask = np.ones((11, 11))

        # Monster feature constants
        # 怪物特征常量
        self.max_monster_count = 2

        # Temporal feature configuration
        # 时序特征配置
        self.temporal_history_len = 5
        self.temporal_state = TemporalState(history_len=self.temporal_history_len)

        # Recent position tracking for reward computation
        # 最近位置跟踪，用于奖励计算
        self.recent_position_map = defaultdict(int)
        self.recent_position_max = 200
        self.recent_positions = deque(maxlen=self.recent_position_max)

        self.reset()

    def reset(self):
        """Reset all states.

        重置所有状态。
        """
        self.cur_pos = (0, 0)
        self.cur_pos_norm = np.array((0, 0))

        self.last_pos_norm = np.array((0, 0))
        self.last_action = -1
        self.bad_move_ids = set()
        self.map_data = np.zeros((2, self.grid_map_size, self.grid_map_size)).astype(int)
        self.map_data[0] = np.ones_like(self.map_data[0])

        self.visited_norm = 0
        self.last_visited_norm = 0
        self.visited_count = 0
        self.last_visited_count = 0
        self.speed_up = False
        self.buff_found = False
        self.buff_status = 0
        self.buff_pos = (0, 0)
        self.buff_cooldown = 0

        self.feature_treasures_pos = [self._get_pos_feature(False, (0, 0), (0, 0))] * 8
        self.treasure_list = []
        self.last_treasure_count = 0
        self.last_buff_count = 0
        self.ray_dist = [0] * 8

        self.step_no = 0
        self.step_no_norm = 0
        self.frame_no = 0
        self.frame_no_norm = 0
        self.move_usable = True
        self.skill_usable = False
        self.last_skill_usable = False
        self.skill_cd = 0

        # Monster states
        # 怪物相关状态
        self.all_monsters = []
        self.monsters = []
        self.monster_count = 0
        self.min_monster_distance = 5
        self.last_min_monster_distance = 5
        self.last_monster_positions = {}

        # v4 dual-monster state / v4 双怪包夹状态
        self.monster_geo = []
        self.dual_safety = 1.0
        self.last_dual_safety = 1.0
        self.monster_spread = 0.0
        self.last_monster_spread = 0.0
        # v4 buff approach tracking / v4 buff 靠近跟踪
        self.buff_dist = 999.0
        self.last_buff_dist = 999.0

        # Temporal state
        # 时序状态
        self.temporal_state.reset()

        # Environment info
        # 环境信息
        self.env_info = {}

        # Recent position tracking
        # 最近位置跟踪
        self.recent_position_map = defaultdict(int)
        self.recent_positions = deque(maxlen=self.recent_position_max)

    def update_map(self, map_data, view, pos):
        """Update map data with the given view at position.

        使用给定位置的视野更新地图数据。
        """
        view_size = view.shape[0]
        l = max(0, pos[0] - view_size // 2)
        r = min(self.grid_map_size, pos[0] + view_size // 2 + 1)
        u = max(0, pos[1] - view_size // 2)
        d = min(self.grid_map_size, pos[1] + view_size // 2 + 1)

        view_x_start = l - (pos[0] - view_size // 2)
        view_x_end = view_x_start + (r - l)
        view_y_start = u - (pos[1] - view_size // 2)
        view_y_end = view_y_start + (d - u)

        cropped_view = view[view_x_start:view_x_end, view_y_start:view_y_end]
        before = np.sum(map_data[l:r, u:d])
        map_data[l:r, u:d] = cropped_view
        after = np.sum(map_data[l:r, u:d])
        return after - before

    def _get_pos_feature(self, found, cur_pos, target_pos):
        """Compute position feature relative to current position (6D).

        计算目标位置相对于当前位置的特征（6维）。
        """
        relative_pos = tuple(y - x for x, y in zip(cur_pos, target_pos))
        dist = np.linalg.norm(relative_pos)
        target_pos_norm = norm_val(target_pos, 128, -128)
        feature = np.array(
            (
                found,
                norm_val(relative_pos[0] / max(dist, 1e-4), 1, -1),
                norm_val(relative_pos[1] / max(dist, 1e-4), 1, -1),
                target_pos_norm[0],
                target_pos_norm[1],
                norm_val(dist, 1.41 * 128) if found else 1.0,
            ),
        )
        return feature

    def _get_monster_feature_v2(self, monster, hero_pos):
        """Enhanced monster feature (10D).

        增强版怪物特征（10维）。

        Features: exist(1) + precise_dist(1) + dist_bucket(1) + direction(2) + velocity(2) + danger(1) + predicted_pos(2)
        特征: 存在(1) + 精确距离(1) + 距离桶(1) + 方向(2) + 速度(2) + 危险(1) + 预测位置(2)

        When monster is None, the slot has no monster.
        当 monster 为 None 时，表示该怪物槽位无怪物。
        When is_in_view=0, the monster exists but is out of view.
        当 is_in_view=0 时，表示怪物存在但不在视野内。
        """
        if monster is None:
            return np.array([0, 1, 1, 0.5, 0.5, 0.5, 0.5, 0, 0.5, 0.5])

        if monster.get("is_in_view", 0) == 0:
            return np.array([0.5, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        exist_flag = 1

        dist_bucket = monster["hero_l2_distance"]
        dist_bucket_norm = dist_bucket / 5.0

        bucket_centers = [15, 45, 75, 105, 135, 165]
        estimated_dist = bucket_centers[min(dist_bucket, 5)]
        precise_dist_norm = norm_val(estimated_dist, 180, 0)

        direction = monster["hero_relative_direction"]
        direction_vectors = {
            0: (0, 0),
            1: (1, 0),
            2: (1, -1),
            3: (0, -1),
            4: (-1, -1),
            5: (-1, 0),
            6: (-1, 1),
            7: (0, 1),
            8: (1, 1),
        }
        dx, dy = direction_vectors.get(direction, (0, 0))
        dir_x_norm = norm_val(dx, 1, -1)
        dir_y_norm = norm_val(dy, 1, -1)

        monster_id = monster.get("monster_id", 0)
        monster_pos = (monster["pos"]["x"], monster["pos"]["z"])

        vx, vy = 0, 0
        if monster_id in self.last_monster_positions:
            last_pos = self.last_monster_positions[monster_id]
            vx = monster_pos[0] - last_pos[0]
            vy = monster_pos[1] - last_pos[1]
            vx = np.clip(vx / 4.0, -1, 1)
            vy = np.clip(vy / 4.0, -1, 1)

        vel_x_norm = norm_val(vx, 1, -1)
        vel_y_norm = norm_val(vy, 1, -1)

        self.last_monster_positions[monster_id] = monster_pos

        dist_danger = 1.0 - dist_bucket_norm
        if abs(vx) > 0.01 or abs(vy) > 0.01:
            hero_dir_x = hero_pos[0] - monster_pos[0]
            hero_dir_y = hero_pos[1] - monster_pos[1]
            hero_dist = max(np.sqrt(hero_dir_x**2 + hero_dir_y**2), 1e-6)
            hero_dir_x /= hero_dist
            hero_dir_y /= hero_dist
            vel_magnitude = max(np.sqrt(vx**2 + vy**2), 1e-6)
            dot_product = vx / vel_magnitude * hero_dir_x + vy / vel_magnitude * hero_dir_y
            approach_factor = (dot_product + 1) / 2
        else:
            approach_factor = 0.5

        danger_level = dist_danger * 0.7 + approach_factor * 0.3

        pred_monster_x = monster_pos[0] + vx * 4 * 5
        pred_monster_y = monster_pos[1] + vy * 4 * 5
        pred_offset_x = pred_monster_x - hero_pos[0]
        pred_offset_y = pred_monster_y - hero_pos[1]
        pred_x_norm = norm_val(pred_offset_x, 128, -128)
        pred_y_norm = norm_val(pred_offset_y, 128, -128)

        return np.array(
            [
                exist_flag,
                precise_dist_norm,
                dist_bucket_norm,
                dir_x_norm,
                dir_y_norm,
                vel_x_norm,
                vel_y_norm,
                danger_level,
                pred_x_norm,
                pred_y_norm,
            ]
        )

    def _get_progress_feature(self, env_info, frame_no, max_step=1000):
        """Progress feature (12D).

        进度特征（12维）。
        """
        monster_interval = env_info.get("monster_interval")
        monster_speed_boost_step = 500
        treasure_count_total = env_info.get("total_treasure")

        remaining_steps = max(0, max_step - frame_no)
        remaining_steps_norm = remaining_steps / max_step
        used_steps_ratio = min(frame_no / max_step, 1.0)

        if monster_interval <= 0 or frame_no >= monster_interval:
            second_monster_countdown = 0
        else:
            second_monster_countdown = (monster_interval - frame_no) / monster_interval

        if frame_no >= monster_speed_boost_step:
            speed_boost_countdown = 0
        else:
            speed_boost_countdown = (monster_speed_boost_step - frame_no) / monster_speed_boost_step

        flash_available = 1 if self.skill_usable else 0
        monster_count_norm = self.monster_count / 2.0

        treasure_collected = env_info.get("treasures_collected")
        if treasure_count_total > 0:
            treasure_progress = treasure_collected / treasure_count_total
        else:
            treasure_progress = 1.0

        remaining_treasures = max(0, treasure_count_total - treasure_collected)
        remaining_treasures_norm = remaining_treasures / max(treasure_count_total, 1)
        buff_available = 1 if (self.buff_found and self.buff_status) else 0
        buff_active = 1 if self.speed_up else 0
        second_half = 1 if frame_no > max_step // 2 else 0
        danger_phase = 1 if frame_no >= monster_speed_boost_step else 0

        return np.array(
            [
                remaining_steps_norm,
                used_steps_ratio,
                second_monster_countdown,
                speed_boost_countdown,
                flash_available,
                monster_count_norm,
                treasure_progress,
                remaining_treasures_norm,
                buff_available,
                buff_active,
                second_half,
                danger_phase,
            ]
        )

    def _get_temporal_feature(self, hero_pos: Tuple[int, int]) -> np.ndarray:
        """Temporal feature (24D): monster history relative positions (20D) + velocity trend (4D).

        时序特征（24维）：怪物历史相对位置（20维）+ 速度变化趋势（4维）。
        """
        features = []

        for monster_idx in range(self.max_monster_count):
            monster_id = None
            if monster_idx < len(self.monsters):
                monster_id = self.monsters[monster_idx].get("monster_id", monster_idx)

            trajectory = self.temporal_state.get_monster_trajectory(monster_id) if monster_id is not None else []
            while len(trajectory) < self.temporal_history_len:
                trajectory.insert(0, (0, 0, 0, 0))
            trajectory = trajectory[-self.temporal_history_len :]

            for __, (mx, mz, vx, vz) in enumerate(trajectory):
                if monster_id is not None and (mx != 0 or mz != 0):
                    rel_x = mx - hero_pos[0]
                    rel_z = mz - hero_pos[1]
                    rel_x_norm = norm_val(rel_x, 128, -128)
                    rel_z_norm = norm_val(rel_z, 128, -128)
                else:
                    rel_x_norm = 0.5
                    rel_z_norm = 0.5
                features.extend([rel_x_norm, rel_z_norm])

            if monster_id is not None:
                ax, az = self.temporal_state.get_velocity_trend(monster_id)
                ax_norm = norm_val(ax, 2, -2)
                az_norm = norm_val(az, 2, -2)
            else:
                ax_norm = 0.5
                az_norm = 0.5
            features.extend([ax_norm, az_norm])

        return np.array(features)

    def _get_path_planning_feature(self, hero_pos: Tuple[int, int], treasure_list: List[dict]) -> np.ndarray:
        """Path planning feature (16D).

        路径规划特征（16维）。
        """
        features = []
        map_data = self.map_data[0]

        escape_scores = calc_escape_routes(map_data, hero_pos, self.monsters)
        features.extend(escape_scores)

        escape_options, congestion = detect_dead_ends(map_data, hero_pos)
        escape_options_norm = escape_options / 8.0
        features.append(escape_options_norm)

        safe_zone, safe_dist = find_nearest_safe_zone(map_data, hero_pos, self.monsters)
        safe_dist_norm = min(1.0, safe_dist / 30.0) if safe_zone else 1.0
        features.append(1.0 - safe_dist_norm)

        treasure_safe_paths = [0.0] * 5
        if treasure_list:
            treasures_with_dist = []
            for t in treasure_list:
                tx, tz = t["pos"]["x"], t["pos"]["z"]
                dist = np.sqrt((hero_pos[0] - tx) ** 2 + (hero_pos[1] - tz) ** 2)
                treasures_with_dist.append((t, dist))
            treasures_with_dist.sort(key=lambda x: x[1])
            for i, (t, __) in enumerate(treasures_with_dist[:5]):
                treasure_pos = (t["pos"]["x"], t["pos"]["z"])
                is_safe = check_treasure_safe_path(map_data, hero_pos, treasure_pos, self.monsters)
                treasure_safe_paths[i] = 1.0 if is_safe else 0.0
        features.extend(treasure_safe_paths)

        features.append(congestion)
        return np.array(features)

    def _get_treasure_priority_feature(self, hero_pos: Tuple[int, int], treasure_list: List[dict]) -> np.ndarray:
        """Treasure priority feature (8D).

        宝箱优先级特征（8维）。
        """
        features = []
        map_data = self.map_data[0]

        priorities = calc_treasure_priority(hero_pos, treasure_list, self.monsters, map_data)

        top3_scores = [0.0, 0.0, 0.0]
        for i, (idx, score) in enumerate(priorities[:3]):
            top3_scores[i] = score
        features.extend(top3_scores)

        safe_windows = [0.0, 0.0, 0.0]
        if treasure_list and self.monsters:
            for i, (idx, __) in enumerate(priorities[:3]):
                if idx < len(treasure_list):
                    t = treasure_list[idx]
                    tx, tz = t["pos"]["x"], t["pos"]["z"]
                    hero_to_treasure = np.sqrt((hero_pos[0] - tx) ** 2 + (hero_pos[1] - tz) ** 2)

                    min_monster_to_treasure = float("inf")
                    for m in self.monsters:
                        mx, mz = m["pos"]["x"], m["pos"]["z"]
                        dist = np.sqrt((tx - mx) ** 2 + (tz - mz) ** 2)
                        min_monster_to_treasure = min(min_monster_to_treasure, dist)

                    monster_speed = 1.5
                    if min_monster_to_treasure > hero_to_treasure:
                        window = (min_monster_to_treasure - hero_to_treasure) / monster_speed
                        safe_windows[i] = min(1.0, window / 50)
        elif treasure_list:
            for i in range(min(3, len(priorities))):
                safe_windows[i] = 1.0
        features.extend(safe_windows)

        if priorities and treasure_list:
            best_idx = priorities[0][0]
            if best_idx < len(treasure_list):
                t = treasure_list[best_idx]
                tx, tz = t["pos"]["x"], t["pos"]["z"]
                dx = tx - hero_pos[0]
                dz = tz - hero_pos[1]
                dist = max(np.sqrt(dx**2 + dz**2), 1e-6)
                dir_x = norm_val(dx / dist, 1, -1)
                dir_z = norm_val(dz / dist, 1, -1)
                features.extend([dir_x, dir_z])
            else:
                features.extend([0.5, 0.5])
        else:
            features.extend([0.5, 0.5])

        return np.array(features)

    def _get_safety_heatmap_feature(self) -> np.ndarray:
        """Safety heatmap feature (16D).

        安全热力图特征（16维）。
        """
        map_data = self.map_data[0]
        heatmap = calc_safety_heatmap(map_data, self.cur_pos, self.monsters, grid_size=4)
        return heatmap.flatten()

    # ================= Auxiliary task labels / 辅助任务标签 =================

    def _get_monster_pred_label(self) -> np.ndarray:
        """Monster position prediction label (8D).

        怪物位置预测标签（8维）。
        """
        labels = []
        for monster_idx in range(self.max_monster_count):
            if monster_idx < len(self.all_monsters) and self.all_monsters[monster_idx].get("is_in_view", 0) == 1:
                monster = self.all_monsters[monster_idx]
                monster_id = monster.get("monster_id", monster_idx)
                monster_pos = (monster["pos"]["x"], monster["pos"]["z"])

                rel_x = monster_pos[0] - self.cur_pos[0]
                rel_z = monster_pos[1] - self.cur_pos[1]

                if monster_id in self.last_monster_positions:
                    last_pos = self.last_monster_positions[monster_id]
                    dx = monster_pos[0] - last_pos[0]
                    dz = monster_pos[1] - last_pos[1]

                    last_hero_pos = (self.last_pos_norm[0] * 256 - 128, self.last_pos_norm[1] * 256 - 128)
                    last_dist = np.sqrt((last_pos[0] - last_hero_pos[0]) ** 2 + (last_pos[1] - last_hero_pos[1]) ** 2)
                    cur_dist = np.sqrt(rel_x**2 + rel_z**2)
                    dist_change = cur_dist - last_dist

                    if abs(dx) < 0.01 and abs(dz) < 0.01:
                        direction = 0.5
                    else:
                        angle = np.arctan2(dz, dx)
                        dir_idx = int(round((angle + np.pi) / (2 * np.pi) * 8)) % 8
                        direction = dir_idx / 7.0
                else:
                    dx, dz = 0, 0
                    dist_change = 0
                    direction = 0.5

                dx_norm = np.clip((dx + 10) / 20, 0, 1)
                dz_norm = np.clip((dz + 10) / 20, 0, 1)
                dist_change_norm = np.clip((dist_change + 20) / 40, 0, 1)

                labels.extend([dx_norm, dz_norm, dist_change_norm, direction])
            else:
                labels.extend([0.5, 0.5, 0.5, 0.5])

        return np.array(labels)

    def _get_safety_pred_label(self) -> np.ndarray:
        """Safety zone classification label (4D).

        安全区域分类标签（4维）。
        """
        map_data = self.map_data[0]
        heatmap = calc_safety_heatmap(map_data, self.cur_pos, self.monsters, grid_size=4)

        quadrant_safety = np.array(
            [
                heatmap[0:2, 0:2].mean(),
                heatmap[0:2, 2:4].mean(),
                heatmap[2:4, 0:2].mean(),
                heatmap[2:4, 2:4].mean(),
            ]
        )
        return quadrant_safety

    def _get_action_hint_label(self) -> np.ndarray:
        """Action direction hint label (8D).

        动作方向建议标签（8维）。
        """
        map_data = self.map_data[0]
        escape_scores = calc_escape_routes(map_data, self.cur_pos, self.monsters)
        return escape_scores

    def _get_aux_labels(self) -> np.ndarray:
        """Get complete auxiliary task labels (20D).

        获取完整的辅助任务标签（20维）。
        """
        monster_pred_label = self._get_monster_pred_label()
        safety_pred_label = self._get_safety_pred_label()
        action_hint_label = self._get_action_hint_label()
        return np.concatenate([monster_pred_label, safety_pred_label, action_hint_label])

    # ================= Reward / 奖励 =================

    def _calc_phase_reward(self, frame_no: int, max_step: int = 1000) -> Dict[str, float]:
        """Compute phased reward weights.

        计算阶段性奖励权重。
        """
        phase1_end = 300
        phase2_end = 500

        if frame_no < phase1_end:
            return {"collect": 1.2, "survive": 0.8, "explore": 1.0}
        elif frame_no < phase2_end:
            return {"collect": 1.0, "survive": 1.0, "explore": 0.8}
        else:
            return {"collect": 0.7, "survive": 1.3, "explore": 0.5}

    @staticmethod
    def _dir_to_angle(direction: int) -> float:
        """Map official relative direction (1-8) to an angle in radians.
        将官方相对方向(1-8)映射为弧度角。
        """
        mapping = {
            1: 0.0,
            2: np.pi / 4,
            3: np.pi / 2,
            4: 3 * np.pi / 4,
            5: np.pi,
            6: -3 * np.pi / 4,
            7: -np.pi / 2,
            8: -np.pi / 4,
            0: 0.0,
        }
        return mapping.get(int(direction), 0.0)

    def _extract_monster_geometry(self):
        """Extract distance bucket and direction angle of visible monsters.
        提取视野内怪物的距离桶与方向角，按距离排序（桶小=近）。
        """
        geo = []
        for m in self.monsters:
            bucket = m["hero_l2_distance"]
            ang = self._dir_to_angle(m.get("hero_relative_direction", 0))
            geo.append((bucket, ang))
        geo.sort(key=lambda g: g[0])
        return geo

    def _calc_dual_safety(self, geo):
        """Composite safety from the nearest-monster bucket and the angle between two monsters.
        基于最近怪距离桶与两怪夹角计算综合安全度（spread 越大=对面包夹越危险）。
        """
        if not geo:
            return 1.0, 0.0
        near_safety = geo[0][0] / 5.0  # bucket 0-5 -> 0-1
        spread = 0.0
        if len(geo) >= 2:
            ang_diff = abs(geo[0][1] - geo[1][1])
            if ang_diff > np.pi:
                ang_diff = 2 * np.pi - ang_diff
            spread = float(ang_diff / np.pi)  # 0 same direction, 1 opposite / 0同向,1对面包夹
        safety = near_safety * (1.0 - 0.4 * spread)
        return float(safety), spread

    def _calc_flash_landing_reward(self) -> float:
        """v4 flash reward: evaluate landing safety vs origin against BOTH monsters,
        landing dead-end, and timing (save the skill for the dual-monster phase).
        v4 穿梭奖励：考核落点相对两怪的综合安全度、落点死角与使用时机（为双怪阶段保留技能）。
        """
        if not (8 <= self.last_action <= 15):
            return 0.0

        safety_gain = self.dual_safety - self.last_dual_safety
        escape_options, _ = detect_dead_ends(self.map_data[0], self.cur_pos)
        is_dual_phase = self.frame_no >= 300 or self.monster_count >= 2

        if is_dual_phase:
            # Dual-monster phase: reward a successful escape to an open landing.
            # 双怪阶段：成功落到开阔安全区重奖。
            if safety_gain > 0 and escape_options >= 3:
                return min(1.0, 0.4 + 2.0 * safety_gain)
            if escape_options < 3:
                return -0.3  # landed in a dead end / 落点死角
            if safety_gain < 0:
                return -0.2  # landing is more dangerous / 落点更危险
            return 0.0
        else:
            # Single-monster phase: save the skill; allow only when in immediate danger.
            # 单怪阶段：保留技能，仅在近身危急时允许使用。
            if self.last_min_monster_distance <= 1 and safety_gain > 0:
                return 0.2
            return -0.1

    def _calc_survive_reward(self, min_dist, last_dist):
        """Compute non-linear safety distance reward.

        计算非线性安全距离奖励。
        """
        survive_reward = 0.02

        if min_dist == 0:
            survive_reward *= 0.5
        elif min_dist == 1:
            survive_reward *= 0.75
        elif min_dist == 2:
            pass
        elif min_dist >= 3:
            survive_reward += 0.01 * (min_dist - 2)

        distance_change = min_dist - last_dist
        if distance_change > 0:
            survive_reward += 0.02 * distance_change
        elif distance_change < 0 and min_dist <= 2:
            survive_reward += 0.01 * distance_change

        return survive_reward

    def _calc_treasure_timing_reward(self, treasure_collected: int, min_dist: int) -> float:
        """Compute treasure collection timing reward.

        计算宝箱收集时机奖励。

        Args:
            treasure_collected: number of treasures collected this step / 本步收集的宝箱数量
            min_dist: current min distance to monster / 当前与怪物的最小距离桶

        Returns:
            float: timing reward value / 时机奖励值
        """
        if treasure_collected <= 0:
            return 0.0

        # 宝箱收集时机奖励：怪物较远时收集给予更高奖励，怪物贴近时收集给予负奖励。
        # min_dist 为距离桶（0-5，越小越危险）。
        if min_dist >= 4:
            return 1.0   # 非常安全，怪物较远或无可见怪物
        elif min_dist == 3:
            return 0.6   # 较安全
        elif min_dist == 2:
            return 0.2   # 有一定风险
        elif min_dist == 1:
            return -0.4  # 危险，怪物贴近
        else:            # min_dist == 0
            return -0.8  # 极度危险

    def update_position(self, hero_grid_pos):
        """Update recent position record for reward computation.

        更新最近位置记录，用于奖励计算。
        """
        if len(self.recent_positions) == self.recent_position_max:
            oldest_pos = self.recent_positions.popleft()
            self.recent_position_map[oldest_pos] -= 1
            if self.recent_position_map[oldest_pos] == 0:
                del self.recent_position_map[oldest_pos]
        self.recent_positions.append(hero_grid_pos)
        self.recent_position_map[hero_grid_pos] += 1

    def pb2struct(self, env_obs):
        """Parse structured information from observation.

        从 obs 解析结构化信息。
        """
        obs = env_obs["observation"]
        frame_state = env_obs["observation"]["frame_state"]

        hero = frame_state["heroes"]

        self.last_pos_norm = self.cur_pos_norm

        self.step_no = obs["step_no"]
        self.step_no_norm = norm_val(self.step_no, 2000)
        self.frame_no = self.step_no
        self.frame_no_norm = norm_val(self.frame_no, 1000 * 3)

        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])
        self.cur_pos_norm = norm_val(self.cur_pos, 128, -128)

        self.move_usable = True

        # Flash skill status
        # 空间穿梭技能状态
        legal_act = obs["legal_action"]
        self.last_skill_usable = self.skill_usable
        if legal_act:
            self.legal_act = legal_act
            self.skill_usable = any(legal_act[8:16])
        self.speed_up = hero["buff_remaining_time"] > 0
        self.skill_cd = hero["flash_cooldown"]

        # Map information
        # 地图信息
        if "map_info" in obs and obs["map_info"]:
            map_info = obs["map_info"]
            if isinstance(map_info[0], dict):
                obstacle_view = np.array([x["values"] for x in map_info])
            else:
                obstacle_view = np.array(map_info)
            self.update_map(self.map_data[0], obstacle_view, self.cur_pos)
        self.update_map(self.map_data[1], self.visit_view_mask, self.cur_pos)
        self.last_visited_norm = self.visited_norm
        self.visited_norm = self.map_data[1].mean()
        self.last_visited_count = self.visited_count
        self.visited_count = int(self.map_data[1].sum())

        # Buff information
        # buff信息
        self.buff_found = False
        self.buff_status = 0
        self.buff_cooldown = 0
        self.buff_pos = self.cur_pos

        organs = frame_state["organs"]
        self.treasure_list = []
        for organ in organs:
            if organ["sub_type"] == 2 and organ.get("status", -1) != -1:
                self.buff_found = True
                self.buff_status = organ["status"] == 1
                self.buff_cooldown = organ.get("cooldown", 0)
                self.buff_pos = (organ["pos"]["x"], organ["pos"]["z"])
            elif organ["sub_type"] == 1 and organ["status"] == 1:
                self.treasure_list.append(organ)

        self.feature_buff_pos = self._get_pos_feature(self.buff_found, self.cur_pos, self.buff_pos)

        # v4 buff approach distance / v4 记录到 buff 的距离（供靠近奖励）
        self.last_buff_dist = self.buff_dist
        if self.buff_found:
            self.buff_dist = float(
                np.hypot(self.buff_pos[0] - self.cur_pos[0], self.buff_pos[1] - self.cur_pos[1])
            )
        else:
            self.buff_dist = 999.0

        # Treasure feature
        # 宝箱特征
        self.feature_treasures_pos = [
            self._get_pos_feature(True, self.cur_pos, (x["pos"]["x"], x["pos"]["z"])) for x in self.treasure_list
        ] + [self._get_pos_feature(False, self.cur_pos, self.cur_pos)] * 13

        self.feature_treasures_pos = self.feature_treasures_pos[:13]
        self.feature_treasures_pos = sorted(self.feature_treasures_pos, key=lambda x: x[-1])

        # Treasure collection count
        # 宝箱收集计数
        self.treasure_get = obs["env_info"]["treasures_collected"] - self.last_treasure_count
        self.last_treasure_count = obs["env_info"]["treasures_collected"]

        self.buff_get = obs["env_info"]["collected_buff"] - self.last_buff_count
        self.last_buff_count = obs["env_info"]["collected_buff"]

        # Monster information: separate all monsters from visible ones
        # 怪物信息：区分全部怪物和视野内怪物
        all_monsters = frame_state.get("monsters", [])
        self.all_monsters = all_monsters
        self.monsters = [m for m in all_monsters if m.get("is_in_view", 0) == 1]
        self.monster_count = len(all_monsters)

        self.last_min_monster_distance = self.min_monster_distance
        if self.monsters:
            self.min_monster_distance = min(m["hero_l2_distance"] for m in self.monsters)
        else:
            self.min_monster_distance = 5

        # v4 dual-monster geometry and composite safety / v4 双怪几何与综合安全度
        self.last_dual_safety = self.dual_safety
        self.last_monster_spread = self.monster_spread
        self.monster_geo = self._extract_monster_geometry()
        self.dual_safety, self.monster_spread = self._calc_dual_safety(self.monster_geo)

        # Environment info
        # 环境信息
        self.env_info = obs.get("env_info")

        # Update temporal state (only use visible monsters, invisible positions are invalid)
        # 更新时序状态（仅使用视野内怪物，视野外位置无效）
        self.temporal_state.update(self.monsters, self.cur_pos, self.last_action)

        # Update recent position record
        # 更新最近位置记录
        hero_grid_pos = (hero["pos"]["x"], hero["pos"]["z"])
        self.update_position(hero_grid_pos)

    def feature_process(self, obs, last_action):
        """Core feature processing method - generates 480D feature + 20D auxiliary labels.

        特征处理核心方法 - 生成480维特征 + 20维辅助标签。

        Returns:
            feature: 480D feature vector / 480维特征向量
            legal_action: legal action mask / 合法动作掩码
            reward: [step_reward, treasure_reward, survive_reward]
            aux_labels: 20D auxiliary task labels / 20维辅助任务标签
        """
        self.last_action = last_action
        self.pb2struct(obs)

        # Ray information
        # 射线信息
        map_data = self.map_data[0]
        delta = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
        max_ray = 10
        self.ray_dist = []
        for dx, dy in delta:
            x, y = self.cur_pos
            cnt = -1
            while (cnt == -1) or (
                x >= 0
                and x < self.grid_map_size
                and y >= 0
                and y < self.grid_map_size
                and int(map_data[x][y]) != 0
                and cnt < max_ray
            ):
                x += dx
                y += dy
                cnt += 1
            self.ray_dist.append(cnt)

        # Flash information
        # 空间穿梭信息
        flash_dist = []
        max_flash_dist = 8000 // 500
        for dx, dy in delta:
            ex, ey = self.cur_pos
            ex += dx * max_flash_dist
            ey += dy * max_flash_dist
            cnt = max_flash_dist
            while (cnt == -1) or (
                ex >= 0
                and ex < self.grid_map_size
                and ey >= 0
                and ey < self.grid_map_size
                and int(map_data[ex][ey]) == 0
                and cnt > 0
            ):
                ex -= dx
                ey -= dy
                cnt -= 1
            flash_dist.append(cnt)

        # Legal action
        # 合法动作
        legal_action = self.get_legal_action()

        # Monster features v2 (10D per monster, max 2)
        # Uses all_monsters for slot alignment; out-of-view monsters get placeholder features
        # 怪物特征 v2（每只怪物10维，最多2只）
        # 使用 all_monsters 保证槽位对齐，不在视野内的怪物返回占位特征
        monster_features_v2 = []
        for i in range(self.max_monster_count):
            if i < len(self.all_monsters):
                monster_features_v2.append(self._get_monster_feature_v2(self.all_monsters[i], self.cur_pos))
            else:
                monster_features_v2.append(self._get_monster_feature_v2(None, self.cur_pos))

        # Progress feature (12D)
        # 进度特征（12维）
        progress_feature = self._get_progress_feature(self.env_info, self.frame_no)

        # Temporal feature (24D)
        # 时序特征（24维）
        temporal_feature = self._get_temporal_feature(self.cur_pos)

        # Path planning feature (16D)
        # 路径规划特征（16维）
        path_planning_feature = self._get_path_planning_feature(self.cur_pos, self.treasure_list)

        # Treasure priority feature (8D)
        # 宝箱优先级特征（8维）
        treasure_priority_feature = self._get_treasure_priority_feature(self.cur_pos, self.treasure_list)

        # Safety heatmap feature (16D)
        # 安全热力图特征（16维）
        safety_heatmap_feature = self._get_safety_heatmap_feature()

        # Concatenate 480D feature
        # 拼接480维特征
        feature = np.concatenate(
            self.feature_treasures_pos  # Treasure position 6*13=78D / 宝箱位置78维
            + [
                self.feature_buff_pos,  # Buff position 6D / buff位置6维
                np.array(
                    (
                        self.buff_status,
                        norm_val(self.buff_cooldown, 100),
                    ),
                ),  # Buff status 2D / buff状态2维
            ]
            + monster_features_v2  # Monster features 10*2=20D / 怪物特征20维
            + [
                progress_feature,  # Progress 12D / 进度12维
                temporal_feature,  # Temporal 24D / 时序24维
                path_planning_feature,  # Path planning 16D / 路径规划16维
                treasure_priority_feature,  # Treasure priority 8D / 宝箱优先级8维
                safety_heatmap_feature,  # Safety heatmap 16D / 安全热力图16维
                np.array(legal_action),  # Legal action 16D / 合法动作16维
                norm_val(self.ray_dist, max_ray),  # Ray distance 8D / 射线距离8维
                norm_val(flash_dist, max_flash_dist),  # Flash distance 8D / 空间穿梭距离8维
                self.cur_pos_norm,  # Current position 2D / 当前位置2维
                np.array(
                    (
                        self.speed_up,
                        self.move_usable,
                        self.skill_usable,
                        norm_val(self.skill_cd, 1000),
                        self.frame_no_norm,
                        self.visited_norm,
                        norm_val(self.last_treasure_count, 13),
                        norm_val(self.min_monster_distance, 5),
                    ),
                ),  # Hero + global state 8D / 英雄+全局状态8维
                self.map_data[1].reshape(16, 8, 16, 8).mean(axis=(1, 3)).flatten(),  # Visited map 256D / 已访问地图256维
            ],
        )

        # Compute auxiliary task labels (20D)
        # 计算辅助任务标签（20维）
        aux_labels = self._get_aux_labels()

        return (
            feature,
            legal_action,
            self.reward_process(),
            aux_labels,
        )

    def get_legal_action(self):
        """Get legal action mask.

        获取合法动作掩码。
        """
        if hasattr(self, "legal_act") and self.legal_act:
            legal_action = [int(x) for x in self.legal_act]
        else:
            legal_action = [self.move_usable] * self.move_action_num + [self.skill_usable] * self.skill_action_num

        if (
            abs(self.cur_pos_norm[0] - self.last_pos_norm[0]) < 0.001
            and abs(self.cur_pos_norm[1] - self.last_pos_norm[1]) < 0.001
            and self.last_action > -1
        ):
            self.bad_move_ids.add(self.last_action)
        else:
            self.bad_move_ids = set()

        for move_id in self.bad_move_ids:
            if move_id < len(legal_action):
                legal_action[move_id] = 0

        if sum(legal_action[: self.move_action_num]) == 0:
            self.bad_move_ids = set()
            return [self.move_usable] * self.move_action_num + [self.skill_usable] * self.skill_action_num

        return legal_action

    def reward_process(self):
        """Reward function.

        奖励函数。

        Returns:
            list: [step_reward, treasure_reward, survive_reward]
        """
        step_reward = 0
        survive_reward = 0
        treasure_reward = 0

        phase_weights = self._calc_phase_reward(self.frame_no)

        # Base survival reward
        # 基础生存奖励
        step_reward += 0.015

        # Monster avoidance reward
        # 怪物躲避奖励
        if self.monsters:
            base_survive = self._calc_survive_reward(self.min_monster_distance, self.last_min_monster_distance)
            survive_reward += base_survive * phase_weights["survive"]
        else:
            survive_reward += 0.02

        # Treasure collection reward
        # 宝箱收集奖励：每个宝箱100分，是得分大头，需给予与生存奖励相当的激励，
        # 配合 phase_weights["collect"]（前期高、后期低）引导前期多收集、后期重生存。
        treasure_collect_weight = 2.5
        treasure_reward += treasure_collect_weight * self.treasure_get * phase_weights["collect"]

        # Treasure collection timing reward
        # 宝箱收集时机奖励
        if self.treasure_get > 0:
            timing_reward = self._calc_treasure_timing_reward(self.treasure_get, self.min_monster_distance)
            treasure_reward += timing_reward * phase_weights["collect"]

        # v4 Buff collection reward: speed boost (2 cells/step for 50 steps) is key to
        # kiting; keep the +1.0 pickup reward in the survival channel.
        # v4 buff收集奖励：加速(一步两格、维持50步)是风筝怪物的关键，拾取给+1.0，计入生存通道。
        if self.buff_get > 0:
            survive_reward += 1.0

        # Exploration reward
        # 探索奖励：根据新增访问格点数奖励探索新区域，帮助发现宝箱与安全路线；
        # 早期探索收益高，地图逐渐探索完后自然衰减。
        if hasattr(self, "visited_count"):
            new_visited_count = self.visited_count - self.last_visited_count
            if new_visited_count > 0:
                exploration_reward = min(0.06, new_visited_count * 0.01)
                step_reward += exploration_reward

        # v4 Flash landing reward (dual-monster aware) / v4 穿梭落点奖励（双怪感知）
        if self.last_action >= 0:
            step_reward += self._calc_flash_landing_reward()

        # v4 Pincer penalty/relief: penalize when the two monsters' enclosing angle widens
        # (encirclement tightening), reward when it opens.
        # v4 包夹奖惩：两怪夹角收紧（包夹）惩罚，夹角展开奖励。
        spread_delta = self.monster_spread - self.last_monster_spread
        survive_reward -= 0.1 * spread_delta

        # v4 Buff approach reward: move toward a visible buff only when relatively safe.
        # v4 buff 靠近奖励：仅在相对安全时朝视野内 buff 靠近。
        if self.buff_found and self.min_monster_distance >= 3 and self.last_buff_dist < 900:
            buff_approach = self.last_buff_dist - self.buff_dist
            if buff_approach > 0:
                survive_reward += min(0.04, 0.02 * buff_approach)

        return [step_reward, treasure_reward, survive_reward]
