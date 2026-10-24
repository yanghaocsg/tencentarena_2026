#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

PPO + RND + Auxiliary task algorithm implementation for Gorge Chase.
峡谷追猎 PPO + RND + 辅助任务算法实现。
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model

        self.optimizer = optimizer
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM

        # Auxiliary task loss weights
        # 辅助任务损失权重
        self.aux_monster_loss_weight = 0.1
        self.aux_safety_loss_weight = 0.1
        self.aux_action_hint_loss_weight = 0.1

        self.last_report_monitor_time = 0
        self.train_step = 0

    def learn(self, list_sample_data):
        """Training entry.

        训练入口。
        """
        obs = torch.stack([frame.obs for frame in list_sample_data]).to(self.model.device)
        legal_action_tensor = torch.stack([frame.legal_action for frame in list_sample_data]).to(self.model.device)
        act = torch.stack([frame.act for frame in list_sample_data]).to(self.model.device).view(-1, 1)
        prob = torch.stack([frame.prob for frame in list_sample_data]).to(self.model.device)
        reward = torch.stack([frame.reward for frame in list_sample_data]).to(self.model.device)
        reward_sum = torch.stack([frame.reward_sum for frame in list_sample_data]).to(self.model.device)
        advantage = torch.stack([frame.advantage for frame in list_sample_data]).to(self.model.device)
        value = torch.stack([frame.value for frame in list_sample_data]).to(self.model.device)
        next_value = torch.stack([frame.next_value for frame in list_sample_data]).to(self.model.device)
        aux_labels = torch.stack([frame.aux_labels for frame in list_sample_data]).to(self.model.device)

        # Model setup
        # 模型设置
        self.model.set_train_mode()
        self.optimizer.zero_grad()

        # Forward pass
        # 前向传播
        rst_list = self.model(obs)
        # rst_list: [label_out, value_out, target, predict, monster_pred, safety_pred, action_hint]

        # Compute loss
        # 计算损失
        total_loss, info_list = self._compute_loss(
            obs=obs,
            legal_action=legal_action_tensor,
            old_action=act,
            old_prob=prob,
            reward=reward,
            reward_sum=reward_sum,
            advantage=advantage,
            old_value=value,
            rst_list=rst_list,
            aux_labels=aux_labels,
        )

        results = {}
        results["total_loss"] = total_loss.item()

        total_loss.backward()

        # Gradient clipping
        # 梯度裁剪
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)

        self.optimizer.step()
        self.train_step += 1

        _info_list = []
        for info in info_list:
            if isinstance(info, list):
                _info = [i.detach().cpu().item() if torch.is_tensor(i) else i for i in info]
            else:
                _info = info.detach().mean().cpu().item() if torch.is_tensor(info) else info
            _info_list.append(_info)

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            results["value_loss"] = round(_info_list[1], 2)
            results["policy_loss"] = round(_info_list[2], 2)
            results["entropy_loss"] = round(_info_list[3], 2)
            results["rnd_loss"] = round(_info_list[4], 2)
            results["aux_loss"] = round(_info_list[5], 2)
            results["reward"] = _info_list[-1]

            self.logger.info(
                f"policy_loss: {results['policy_loss']}, value_loss: {results['value_loss']}, "
                f"entropy_loss: {results['entropy_loss']}, aux_loss: {results['aux_loss']}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})

            self.last_report_monitor_time = now

    def _compute_loss(
        self, obs, legal_action, old_action, old_prob, reward, reward_sum, advantage, old_value, rst_list, aux_labels
    ):
        """Compute PPO + RND + auxiliary task loss.

        计算 PPO + RND + 辅助任务损失。
        """
        label_out = rst_list[0]
        value_pred = rst_list[1]
        target = rst_list[2]
        predict = rst_list[3]
        monster_pred = rst_list[4]
        safety_pred = rst_list[5]
        action_hint = rst_list[6]

        # Use reward_sum (GAE td-lambda return) as value target
        # 使用 reward_sum（GAE td-lambda return）作为价值目标
        tdret = reward_sum

        # Value loss (clipped)
        # 价值损失（裁剪）
        value_pred_squeezed = value_pred.squeeze(-1) if value_pred.dim() > 2 else value_pred
        old_value_squeezed = old_value.squeeze(-1) if old_value.dim() > 2 else old_value
        tdret_squeezed = tdret.squeeze(-1) if tdret.dim() > 2 else tdret
        adv_squeezed = advantage.squeeze(-1) if advantage.dim() > 2 else advantage

        value_clip = old_value_squeezed + (value_pred_squeezed - old_value_squeezed).clamp(
            -self.clip_param, self.clip_param
        )
        value_loss1 = torch.square(tdret_squeezed - value_clip)
        value_loss2 = torch.square(tdret_squeezed - value_pred_squeezed)
        value_loss = 0.5 * torch.maximum(value_loss1, value_loss2).mean()

        # Policy loss (PPO clip)
        # 策略损失（PPO裁剪）
        prob_dist = self._masked_softmax(label_out, legal_action)

        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1).mean()

        one_hot_action = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size)
        new_prob = (one_hot_action * prob_dist).sum(1, keepdim=True)

        old_action_prob = (one_hot_action * old_prob).sum(1, keepdim=True)

        ratio = new_prob / old_action_prob.clamp(1e-9)

        if adv_squeezed.dim() > 1:
            adv_for_policy = adv_squeezed.mean(dim=-1, keepdim=True)
        else:
            adv_for_policy = adv_squeezed.unsqueeze(-1)

        # v2: normalize advantages across the batch (standard PPO) to stabilize updates.
        # v2：在batch内标准化优势（PPO标准做法），三头平均后尺度统一，显著降低策略震荡。
        adv_for_policy = (adv_for_policy - adv_for_policy.mean()) / (adv_for_policy.std() + 1e-8)

        policy_loss1 = -ratio * adv_for_policy
        policy_loss2 = -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv_for_policy
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()

        # RND loss
        # RND损失
        rnd_loss = torch.mean((target - predict) ** 2)

        # Auxiliary task loss
        # 辅助任务损失
        # aux_labels: [batch_size, 20] = monster(8) + safety(4) + action_hint(8)
        monster_target = aux_labels[:, :8]
        safety_target = aux_labels[:, 8:12].clamp(0, 1)
        action_hint_target = aux_labels[:, 12:20]

        aux_monster_loss = F.mse_loss(monster_pred, monster_target)
        aux_safety_loss = F.binary_cross_entropy(safety_pred, safety_target)
        aux_action_hint_loss = F.mse_loss(action_hint, action_hint_target)

        aux_loss = (
            self.aux_monster_loss_weight * aux_monster_loss
            + self.aux_safety_loss_weight * aux_safety_loss
            + self.aux_action_hint_loss_weight * aux_action_hint_loss
        )

        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss + rnd_loss + aux_loss

        info_list = [tdret.mean(), value_loss, policy_loss, entropy_loss, rnd_loss, aux_loss]
        info_list += [adv_squeezed.mean(), adv_squeezed.std(), reward.mean()]

        return total_loss, info_list

    def _masked_softmax(self, logits, legal_action):
        """Apply legal_action mask then compute softmax.

        对 logits 应用 legal_action mask 后计算 softmax。
        """
        label_max, _ = torch.max(logits * legal_action, 1, True)
        label = logits - label_max
        label = label * legal_action
        label = label + 1e5 * (legal_action - 1)
        return torch.nn.functional.softmax(label, dim=1)
