# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


# Configuration, including dimension settings and algorithm parameter settings.
# 配置，包含维度设置，算法参数设置
class Config:

    NUMB_HEAD = 1

    # ---- Feature dimensions (480D) ----
    # ---- 特征维度 (480维) ----
    #   treasure(6*13=78), buff(8), monster_v2(10*2=20), progress(12),
    #   temporal(24), path_planning(16), treasure_priority(8), safety_heatmap(16),
    #   dist(16+8+8=32), global(2+8=10), visited(16*16=256)
    FEATURES = [
        6 * 13,  # treasure pos (78D / 78维)
        8,  # buff status + pos (8D / 8维)
        10 * 2,  # monster features v2 (20D: 10D x 2 monsters / 20维: 10维 × 2只怪物)
        12,  # progress features (12D / 12维)
        24,  # temporal features (24D: history 20D + velocity trend 4D / 24维: 历史轨迹20维 + 速度趋势4维)
        16,  # path planning features (16D / 16维)
        8,  # treasure priority features (8D / 8维)
        16,  # safety heatmap features (16D: 4x4 grid / 16维: 4x4区域)
        16 + 8 + 8,  # legal_action + ray_dist + flash_dist (32D / 32维)
        2 + 8,  # cur_pos_norm + hero + global (10D / 10维)
        16 * 16,  # visited map (256D / 256维)
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)

    # Observation dimension
    # 观测维度
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space: 8 directional moves + 8 directional skills
    # 动作空间: 8方向移动 + 8方向技能
    ACTION_NUM = 16

    # Multi-head value (step / treasure / monster avoidance)
    # 多头价值 (步骤/宝箱/怪物躲避)
    VALUE_NUM = 3

    # Auxiliary task label dimension (20D)
    # 辅助任务标签维度 (20维)
    AUX_LABELS_LEN = 20

    # PPO algorithm parameters
    # PPO 算法参数
    GAMMA = 0.99
    LAMDA = 0.95

    # Initial learning rate
    # 初始学习率
    INIT_LEARNING_RATE_START = 0.0003

    # Entropy regularization coefficient
    # 熵正则化系数
    BETA_START = 0.0005

    # Clip parameter
    # 裁剪参数
    CLIP_PARAM = 0.2

    # Value function loss coefficient
    # 价值函数损失的系数
    VF_COEF = 1

    LABEL_SIZE_LIST = [
        ACTION_NUM,
    ]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()

    # Gradient clipping
    # 梯度裁剪
    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5
