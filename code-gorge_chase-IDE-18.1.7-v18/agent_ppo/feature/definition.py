#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definition and GAE computation, including auxiliary task labels.
数据定义与 GAE 计算，包含辅助任务标签字段。
"""


import numpy as np
from common_python.utils.common_func import create_cls, attached
from agent_ppo.conf.conf import Config

ObsData = create_cls("ObsData", feature=None, legal_action=None, sub_action_mask=None)

ActData = create_cls(
    "ActData",
    action=None,
    d_action=None,
    prob=None,
    value=None,
    target=None,
    predict=None,
    monster_pred=None,
    safety_pred=None,
    action_hint=None,
)

# SampleData includes aux_labels for auxiliary tasks (20D)
# SampleData 包含 aux_labels 辅助任务标签 (20维)
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,
    reward_sum=Config.VALUE_NUM,
    done=1,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
    sub_action=1,
    is_train=1,
    target_rnd=256,
    predict_rnd=256,
    aux_labels=Config.AUX_LABELS_LEN,
)


def sample_process(list_sample_data):
    """Process samples: fill next_value and compute GAE.

    样本处理：填充 next_value 并计算 GAE。
    """
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    _calc_reward(list_sample_data)

    return list_sample_data


def _calc_reward(list_sample_data):
    """Compute cumulative reward and advantage using GAE.

    使用 GAE 计算累积奖励和优势。
    """
    gae = 0.0
    gamma, lamda = Config.GAMMA, Config.LAMDA
    for rl_info in reversed(list_sample_data):
        delta = -rl_info.value + rl_info.reward + gamma * rl_info.next_value
        gae = gae * gamma * lamda + delta
        rl_info.advantage = gae
        rl_info.reward_sum = gae + rl_info.value
