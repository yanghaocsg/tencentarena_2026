#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Gorge Chase policy network with Sub-MLP + Attention + Independent Critic + RND + Auxiliary heads.
峡谷追猎策略网络，包含多子MLP + 注意力机制 + 独立Critic + RND + 辅助任务头。

Usage / 使用方式:
    model = Model(device)
    model(s, inference=False)
        s: concatenated feature tensor, shape (B, DIM_OF_OBSERVATION)
        s: 拼接后的特征 tensor, 形状为 (B, DIM_OF_OBSERVATION)

Output / 输出:
    inference=True:  [logits, value, target, predict, monster_pred, safety_pred, action_hint]
    inference=False: [label_result, value, target, predict, monster_pred, safety_pred, action_hint]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List
from agent_ppo.conf.conf import Config


class RNDNetwork(nn.Module):
    """Random Network Distillation for curiosity-driven exploration.

    随机网络蒸馏，用于内在好奇心驱动的探索。
    """

    def __init__(self, input_dim):
        super().__init__()
        self.target_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        for param in self.target_net.parameters():
            param.requires_grad_(False)

        self.predict_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        target = self.target_net(x)
        predict = self.predict_net(x)
        return target, predict


class CriticNetwork(nn.Module):
    """Independent value network (Critic), separated from the Actor to reduce gradient interference.

    独立的价值网络（Critic），与策略网络（Actor）分离，减少梯度干扰。
    """

    def __init__(self, feature_dim: int, value_num: int = 3, hidden_dims: List[int] = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.value_num = value_num

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        backbone_layers = []
        in_dim = feature_dim
        for __, out_dim in enumerate(hidden_dims):
            backbone_layers.append(nn.Linear(in_dim, out_dim))
            backbone_layers.append(nn.ReLU())
            backbone_layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.backbone = nn.Sequential(*backbone_layers)

        self.value_head = nn.Sequential(nn.Linear(hidden_dims[-1], 64), nn.ReLU(), nn.Linear(64, value_num))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

        for m in self.value_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(features)
        value = self.value_head(hidden)
        return value


class AuxiliaryHeads(nn.Module):
    """Auxiliary task heads: monster prediction (8D), safety classification (4D), action hint (8D).

    辅助任务头：怪物位置预测(8维)、安全区域分类(4维)、动作建议(8维)。
    """

    def __init__(self, input_dim: int = 256, num_monsters: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.num_monsters = num_monsters

        self.monster_pred_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_monsters * 4),
        )

        self.safety_pred_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

        self.action_hint_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor):
        monster_pred = self.monster_pred_head(features)
        safety_pred = torch.sigmoid(self.safety_pred_head(features))
        action_hint = self.action_hint_head(features)
        return monster_pred, safety_pred, action_hint


class Model(nn.Module):
    """Gorge Chase attention-based model with Sub-MLP + Attention + Independent Critic + RND + Auxiliary heads.

    峡谷追猎注意力模型，包含子MLP + 注意力 + 独立Critic + RND + 辅助任务头。
    """

    def __init__(self, device=None):
        super(Model, self).__init__()
        self.model_name = "gorge_chase"
        self.device = device

        # Feature configure parameters
        # 特征配置参数
        self.feature_split_shape = Config.FEATURE_SPLIT_SHAPE
        self.feature_len = Config.FEATURE_LEN
        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM

        # Sub-MLP networks for each feature (11 sub-features)
        # 为每个子特征创建子MLP网络（共11个子特征）
        self.treasure_mlp = MLP([6, 64, 64], "treasure_mlp")
        self.buff_mlp = MLP([8, 64, 64], "buff_mlp")
        self.monster_mlp = MLP([10, 64, 64], "monster_mlp")
        self.progress_mlp = MLP([12, 64, 64], "progress_mlp")
        self.temporal_mlp = MLP([24, 64, 64], "temporal_mlp")
        self.path_planning_mlp = MLP([16, 64, 64], "path_planning_mlp")
        self.treasure_priority_mlp = MLP([8, 64, 64], "treasure_priority_mlp")
        self.safety_heatmap_mlp = MLP([16, 64, 64], "safety_heatmap_mlp")
        self.dist_mlp = MLP([16 + 8 + 8, 64, 64], "dist_mlp")
        self.global_mlp = MLP([2 + 8, 64, 64], "global_mlp")
        self.visited_mlp = MLP([16 * 16, 64, 64], "visited_mlp")

        # Standard attention: query from 6 global-aware features
        # 标准注意力机制：Query由6个全局感知特征构成
        self.query_layer = nn.Linear(64 * 6, 64)
        self.key_layer = nn.Linear(64, 64)
        self.value_layer = nn.Linear(64, 64)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        # Main MLP network: 9 x 64D features = 576D
        # 主MLP网络：9个64维特征拼接 = 576维
        self.main_mlp_net = MLP([64 * 9, 512, 256], "main_mlp_net", non_linearity_last=True)

        # Actor head
        # Actor 输出头
        self.label_mlp = MLP([256, 64, self.label_size], "label_mlp")

        # Independent Critic network
        # 独立的Critic网络
        self.critic_net = CriticNetwork(
            feature_dim=64 * 9,
            value_num=self.value_num,
            hidden_dims=[512, 256, 128],
        )

        # Auxiliary task heads
        # 辅助任务头
        self.auxiliary_heads = AuxiliaryHeads(input_dim=256, num_monsters=2)

        # RND network
        # RND 网络
        self.rnd_net = RNDNetwork(self.feature_len)

    def process_legal_action(self, label, legal_action):
        label_max, _ = torch.max(label * legal_action, 1, True)
        label = label - label_max
        label = label * legal_action
        label = label + 1e5 * (legal_action - 1)
        return label

    def _forward_core(self, feature, legal_action=None):
        """Core forward pass.

        核心前向传播。
        """
        # Split features into 11 sub-features
        # 将特征拆分为11个子特征
        (
            treasure_feature,
            buff_feature,
            monster_feature,
            progress_feature,
            temporal_feature,
            path_planning_feature,
            treasure_priority_feature,
            safety_heatmap_feature,
            dist_feature,
            global_feature,
            visited_feature,
        ) = feature.split(self.feature_split_shape, dim=1)

        treasure_feature = treasure_feature.reshape(-1, 13, 6)
        buff_feature = buff_feature.reshape(-1, 1, 8)
        monster_feature = monster_feature.reshape(-1, 2, 10)

        # Process each feature with its MLP
        # 处理每个特征
        treasure_out = self.treasure_mlp(treasure_feature)
        buff_out = self.buff_mlp(buff_feature)
        monster_out = self.monster_mlp(monster_feature)
        progress_out = self.progress_mlp(progress_feature)
        temporal_out = self.temporal_mlp(temporal_feature)
        path_planning_out = self.path_planning_mlp(path_planning_feature)
        treasure_priority_out = self.treasure_priority_mlp(treasure_priority_feature)
        safety_heatmap_out = self.safety_heatmap_mlp(safety_heatmap_feature)
        dist_out = self.dist_mlp(dist_feature)
        global_out = self.global_mlp(global_feature)
        visited_out = self.visited_mlp(visited_feature)

        none_target = torch.zeros_like(monster_out[:, 0:1, :])

        # Query: dist, global, visited, progress, path_planning, safety_heatmap (6 x 64D)
        # Query: 距离、全局、已访问、进度、路径规划、安全热力图（6个64维）
        query = self.query_layer(
            torch.cat([dist_out, global_out, visited_out, progress_out, path_planning_out, safety_heatmap_out], dim=1)
        ).reshape(-1, 1, 64)

        # Key/Value: treasure, buff, monster, temporal
        # Key/Value: 宝箱、buff、怪物、时序
        temporal_reshaped = temporal_out.reshape(-1, 1, 64)
        key = self.key_layer(torch.cat([treasure_out, buff_out, monster_out, temporal_reshaped, none_target], dim=1))
        value = self.value_layer(
            torch.cat([treasure_out, buff_out, monster_out, temporal_reshaped, none_target], dim=1)
        )

        attn_output, __ = self.attention(query, key, value)
        attended_features = attn_output.reshape(-1, 64)

        # Combine all features: 9 x 64D = 576D
        # 结合所有特征：9个64维 = 576维
        combined_features = torch.cat(
            [
                attended_features,
                dist_out,
                global_out,
                visited_out,
                progress_out,
                temporal_out,
                path_planning_out,
                treasure_priority_out,
                safety_heatmap_out,
            ],
            dim=1,
        )

        # Main MLP processing (for Actor)
        # 主MLP处理（用于Actor）
        fc_mlp_out = self.main_mlp_net(combined_features)

        # Action processing (Actor head)
        # 动作处理（Actor头）
        label_mlp_out = self.label_mlp(fc_mlp_out)
        if legal_action is not None:
            label_out = self.process_legal_action(label_mlp_out, legal_action)
        else:
            label_out = label_mlp_out
        prob = torch.nn.functional.softmax(label_out, dim=1)

        # Value processing (Independent Critic network)
        # 价值处理（独立Critic网络）
        value_out = self.critic_net(combined_features)

        # Auxiliary tasks prediction
        # 辅助任务预测
        monster_pred, safety_pred, action_hint = self.auxiliary_heads(fc_mlp_out)

        # RND forward pass
        # RND前向传播
        target, predict = self.rnd_net(feature)

        return prob, value_out, target, predict, label_out, monster_pred, safety_pred, action_hint

    def forward(self, s, inference=False):
        """Forward pass.

        前向传播。

        Args:
            s: concatenated feature tensor, shape (B, DIM_OF_OBSERVATION)
               拼接后的特征 tensor, 形状 (B, DIM_OF_OBSERVATION)
            inference: inference mode flag / 推理模式标志

        Returns:
            inference=True:  [logits, value, target, predict, monster_pred, safety_pred, action_hint]
            inference=False: [label_logits, value, target, predict, monster_pred, safety_pred, action_hint]
        """
        feature = s.to(torch.float32)

        prob, value_out, target, predict, label_out, monster_pred, safety_pred, action_hint = self._forward_core(
            feature
        )

        if inference:
            logits = label_out
            return [logits, value_out, target, predict, monster_pred, safety_pred, action_hint]
        else:
            return [label_out, value_out, target, predict, monster_pred, safety_pred, action_hint]

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()


def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    """Wrapper function to create and initialize a linear layer.

    用于创建和初始化线性层的包装函数。
    """
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)

    # Initialize weight and bias
    # 初始化权重及偏移量
    nn.init.orthogonal_(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)

    return fc_layer


class MLP(nn.Module):
    """A simple multi-layer perceptron.

    一个简单的多层感知器。
    """

    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
