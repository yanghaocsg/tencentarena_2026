#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Curriculum learning schedule for Gorge Chase PPO.
峡谷追猎 PPO 课程学习进度表。

Difficulty ramps by completed episode count (per aisrv worker):
single monster -> two monsters (no speedup, frequent buffs)
-> two monsters + speedup -> full competition config.
难度按已完成局数（每个 aisrv worker 独立计数）递进：
单怪 -> 双怪(不加速、buff高频) -> 双怪+加速 -> 完整比赛配置。
"""

import copy


# Stage boundaries (exclusive, in episodes) and their config overrides.
# All values stay inside the official legal ranges in train_env_conf.toml.
# 各阶段局数上界（不含）与配置覆盖；取值均在 train_env_conf.toml 官方合法范围内。
STAGES = [
    # Stage 0: single monster, learn basic kiting + collecting (0-599).
    # 阶段0：单怪，学习基础风筝与收集。
    (
        600,
        {
            "monster_interval": 2000,  # second monster effectively never appears / 第二怪不出现
            "monster_speedup": 2000,  # no speed boost within an episode / 局内不加速
            "treasure_count": 6,
        },
    ),
    # Stage 1: two monsters appear early, no speedup; buffs refresh fast so the
    # agent practices flash landings, dual-monster kiting and buff pickup (600-1799).
    # 阶段1：双怪早出现、不加速；buff高频刷新，练习穿梭落点、双怪风筝与捡buff。
    (
        1800,
        {
            "monster_interval": 150,  # second monster appears early, longer dual phase / 双怪早出现
            "monster_speedup": 2000,
            "treasure_count": 10,
            "buff_count": 2,
            "buff_cooldown": 50,  # buffs refresh frequently / buff高频刷新
        },
    ),
    # Stage 2: full monster timing including speedup, practice late-game survival (1800-2999).
    # 阶段2：完整怪物节奏（含加速），练习后期生存。
    (
        3000,
        {
            "monster_interval": 300,
            "monster_speedup": 500,
            "treasure_count": 10,
            "buff_count": 2,
            "buff_cooldown": 200,
        },
    ),
    # Stage 3 (>=3000): no overrides -> exactly the base toml (competition config).
    # 阶段3：无覆盖，完全使用基础 toml（比赛配置）精调。
]

STAGE_NAMES = ["0_single", "1_dual_nospeed", "2_dual_speed", "3_full"]


def get_stage_index(episode_cnt: int) -> int:
    """Return the curriculum stage index for the given completed episode count."""
    for i, (end_episode, _) in enumerate(STAGES):
        if episode_cnt < end_episode:
            return i
    return len(STAGES)


def get_stage_name(episode_cnt: int) -> str:
    return STAGE_NAMES[get_stage_index(episode_cnt)]


def build_curriculum_conf(base_conf, episode_cnt):
    """Return the per-episode config with the current stage overrides applied.

    在基础配置上按当前课程阶段覆盖参数，返回本局配置。
    Adapted to either a flat dict or a nested {"env_conf": {...}} structure.
    自动适配扁平 dict 或嵌套 {"env_conf": {...}} 结构。
    """
    stage = get_stage_index(episode_cnt)
    # Final stage: use the base config unchanged.
    # 最终阶段：原样使用基础配置。
    if stage >= len(STAGES):
        return base_conf

    overrides = STAGES[stage][1]
    conf = copy.deepcopy(base_conf)

    if isinstance(conf, dict) and isinstance(conf.get("env_conf"), dict):
        target = conf["env_conf"]
    elif isinstance(conf, dict):
        target = conf
    else:
        return base_conf

    for key, value in overrides.items():
        if key in target:
            target[key] = value
    return conf
