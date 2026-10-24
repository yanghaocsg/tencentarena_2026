#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Gorge Chase Agent class based on kaiwudrl BaseAgent interface.
峡谷追猎 Agent 主类，基于 kaiwudrl BaseAgent 接口。
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import random

import numpy as np
from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model
from kaiwudrl.interface.agent import BaseAgent


def random_choice(p):
    """Sample action from probability distribution.

    按概率分布随机采样动作。
    """
    r = random.random() * sum(p)
    s = 0
    for i in range(len(p)):
        if r > s and r <= s + p[i]:
            return i, p[i]
        s += p[i]
    return len(p) - 1, p[len(p) - 1]


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        initial_lr = Config.INIT_LEARNING_RATE_START
        parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(params=parameters, lr=initial_lr, betas=(0.9, 0.999), eps=1e-8)
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.logger = logger
        self.monitor = monitor
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, self.logger, self.monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1

        # Cumulative reward statistics per category
        # 分类奖励累计统计
        self.step_reward_sum = 0
        self.treasure_reward_sum = 0
        self.survive_reward_sum = 0

        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs):
        """Reset agent state.

        重置 Agent 状态。
        """
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.step_reward_sum = 0
        self.treasure_reward_sum = 0
        self.survive_reward_sum = 0

    def _predict_with_model(self, feature, legal_action):
        """Run model inference without gradient computation.

        使用模型进行无梯度推理。
        """
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature])).view(1, Config.DIM_OF_OBSERVATION).float().to(self.device)

        with torch.no_grad():
            output_list = self.model(obs_tensor, inference=True)

        # Output: [logits, value, target, predict, monster_pred, safety_pred, action_hint]
        # 输出: [logits, value, target, predict, monster_pred, safety_pred, action_hint]
        np_output = []
        for output in output_list:
            np_output.append(output.cpu().numpy())

        logits = np_output[0][0]
        value = np_output[1][0]
        target = np_output[2][0]
        predict = np_output[3][0]
        monster_pred = np_output[4][0]
        safety_pred = np_output[5][0]
        action_hint = np_output[6][0]

        return logits, value, target, predict, monster_pred, safety_pred, action_hint

    def predict(self, list_obs_data):
        """Predict action from observation data.

        根据观测数据推理动作。
        """
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        logits, value, target, predict, monster_pred, safety_pred, action_hint = self._predict_with_model(
            feature, legal_action
        )

        # Apply legal_action mask and softmax
        # 使用 legal_action mask 进行 softmax
        legal_action_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits, legal_action_np)
        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)

        list_act_data = [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
                target=target,
                predict=predict,
                monster_pred=monster_pred,
                safety_pred=safety_pred,
                action_hint=action_hint,
            )
        ]
        return list_act_data

    def exploit(self, env_obs):
        """Evaluation mode inference (greedy).

        评估模式推理（贪心）。
        """
        obs_data, remain_info = self.observation_process(env_obs)
        if not obs_data:
            return 0
        act_data = self.predict([obs_data])
        act = self.action_process(act_data[0], False)
        return act

    def learn(self, list_sample_data):
        """Train the model.

        训练模型。
        """
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """Save model checkpoint.

        保存模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """Load model checkpoint.

        加载模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.model.device))
        self.logger.info(f"load model {model_file_path} successfully")

    def observation_process(self, env_obs):
        """Process observation into 480D feature + auxiliary labels.

        将观测处理为480维特征 + 辅助任务标签。
        """
        feature, legal_action, reward, aux_labels = self.preprocessor.feature_process(env_obs, self.last_action)

        remain_info = {
            "aux_labels": aux_labels,
            "reward": reward,
        }

        return (
            ObsData(
                feature=list(feature),
                legal_action=legal_action,
                sub_action_mask=[
                    1,
                ]
                * Config.NUMB_HEAD,
            ),
            remain_info,
        )

    def action_process(self, act_data, is_stochastic=True):
        """Process action data and update last_action.

        处理动作数据并更新 last_action。
        """
        if is_stochastic:
            action = act_data.action
        else:
            action = act_data.d_action

        self.last_action = int(action[0])
        return int(action[0])

    def _legal_soft_max(self, input_hidden, legal_action):
        """Softmax with legal action masking.

        合法动作掩码下的 softmax。
        """
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5

        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        probs = tmp / (np.sum(tmp, keepdims=True) * 1.00001)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        """Sample action from probability distribution.

        基于概率分布采样动作。
        """
        if use_max:
            return np.argmax(probs)
        return np.argmax(np.random.multinomial(1, probs, size=1))
