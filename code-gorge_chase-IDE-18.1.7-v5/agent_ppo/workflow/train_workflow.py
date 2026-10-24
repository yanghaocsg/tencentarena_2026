#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow with EpisodeRunner for Gorge Chase.
峡谷追猎训练工作流，基于 EpisodeRunner 框架接口。
"""


import os
import time

import numpy as np
from agent_ppo.conf.conf import Config
from agent_ppo.conf.curriculum import build_curriculum_conf, get_stage_name
from agent_ppo.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    # Read and validate configuration
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    # Create EpisodeRunner instance
    # 创建 EpisodeRunner 实例
    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        # Run episodes and collect training data
        # 运行对局并收集数据
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            # Save model checkpoint periodically
            # 定期保存模型文件
            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

    def run_episodes(self):
        """Run a single episode.

        单局流程。
        """
        while True:
            # Get training metrics periodically
            # 定期获取训练指标
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            # v5 Build the per-episode config from the curriculum schedule.
            # v5 按课程进度表生成本局配置。
            stage_name = get_stage_name(self.episode_cnt)
            cur_conf = build_curriculum_conf(self.usr_conf, self.episode_cnt)
            env_obs = self.env.reset(cur_conf)

            # Disaster recovery
            # 容灾
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Load latest model at episode start
            # 每次对局开始时加载最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # Feature processing
            # 特征处理
            obs_data, remain_info = self.agent.observation_process(env_obs)
            obs = env_obs["observation"]

            # Initialize frame collector
            # 初始化帧收集器
            collector = list()

            # Episode variables
            # 对局变量
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0

            self.logger.info(
                f"Episode {self.episode_cnt} stage {stage_name} start, usr_conf is {cur_conf}"
            )

            while not done:
                # Agent inference
                # Agent 推理
                act_data = self.agent.predict(list_obs_data=[obs_data])
                act_data = act_data[0]

                # Unpack action
                # 动作解包
                act = self.agent.action_process(act_data)

                # Interact with environment
                # 与环境交互
                env_reward, env_obs = self.env.step(act)

                # Disaster recovery
                # 容灾
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                frame_no = env_obs["frame_no"]
                _obs = env_obs["observation"]
                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1

                # Feature processing
                # 特征处理
                _obs_data, _remain_info = self.agent.observation_process(env_obs)
                _obs_data.frame_no = frame_no

                total_score = _obs["env_info"]["total_score"]

                # Get 3D reward from remain_info (computed in observation_process)
                # 从 remain_info 获取三维奖励（已在 observation_process 中计算）
                reward = np.array(_remain_info.get("reward", [0, 0, 0]))

                # Compute RND intrinsic reward and add to reward[0]
                # 计算 RND 内在奖励并加到 reward[0]
                if act_data.target is not None and act_data.predict is not None:
                    rnd_reward = np.mean((act_data.target - act_data.predict) ** 2)
                    reward[0] += min(0.1, rnd_reward)

                # Accumulate reward by category
                # 分类统计奖励
                self.agent.step_reward_sum += reward[0]
                self.agent.treasure_reward_sum += reward[1]
                self.agent.survive_reward_sum += reward[2]

                done = terminated or truncated

                # Get auxiliary task labels
                # 获取辅助任务标签
                aux_labels = _remain_info.get("aux_labels", np.zeros(Config.AUX_LABELS_LEN))

                # Check episode end and apply terminal reward bonus
                # 判断环境结束并增强终局奖励
                if done:
                    self.logger.info(f"Episode ended at step {step}, with a total score of {total_score}.")
                    fm = self.agent.preprocessor

                    if terminated:
                        # Caught by monster, fail - apply enhanced penalty
                        # 被怪物抓住，失败 - 增强惩罚
                        survival_factor = min(fm.frame_no_norm + 0.1, 1)
                        base_penalty = -10 * (1 - survival_factor)
                        treasure_bonus = 0.5 * fm.last_treasure_count if survival_factor > 0.5 else 0
                        final_reward_scalar = base_penalty + treasure_bonus
                        final_reward = np.array([0, 0, final_reward_scalar])
                        result_str = "FAIL"
                    elif truncated:
                        # Survived to max steps, win - large positive reward
                        # 存活到最大步数，胜利 - 大额正奖励
                        final_reward_scalar = 20 + 2.0 * fm.last_treasure_count
                        final_reward = np.array([0, 0, final_reward_scalar])
                        result_str = "WIN"
                    else:
                        final_reward = np.zeros(Config.VALUE_NUM)
                        result_str = "ABNORMAL"

                    # Get simulator actual score
                    # 获取 simulator 实际得分
                    env_info = getattr(fm, "env_info", {})
                    sim_total_score = env_info.get("total_score", 0)
                    sim_treasure_score = env_info.get("treasure_score", 0)

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} result:{result_str} "
                        f"final:{final_reward_scalar:.2f} rewards:[step:{self.agent.step_reward_sum:.2f}, "
                        f"treasure:{self.agent.treasure_reward_sum:.2f}, "
                        f"survive:{self.agent.survive_reward_sum:.2f}] "
                        f"treasure_cnt:{fm.last_treasure_count} explore:{fm.visited_norm * 100:.1f}% "
                        f"sim_score:[total:{sim_total_score}, treasure:{sim_treasure_score}]"
                    )

                # Build sample frame
                # 构造样本帧
                total_reward += reward.sum()
                prob = act_data.prob
                value_arr = act_data.value
                action = act_data.action

                target_rnd = act_data.target if act_data.target is not None else np.zeros(256)
                predict_rnd = act_data.predict if act_data.predict is not None else np.zeros(256)

                frame = SampleData(
                    obs=np.array(obs_data.feature),
                    legal_action=np.array(obs_data.legal_action),
                    act=action,
                    reward=reward,
                    done=done,
                    reward_sum=np.zeros(Config.VALUE_NUM),
                    value=value_arr.flatten()[: Config.VALUE_NUM],
                    next_value=np.zeros(Config.VALUE_NUM),
                    advantage=np.zeros(Config.VALUE_NUM),
                    prob=prob,
                    sub_action=np.array(obs_data.sub_action_mask),
                    is_train=1,
                    target_rnd=target_rnd,
                    predict_rnd=predict_rnd,
                    aux_labels=aux_labels,
                )

                collector.append(frame)

                # If episode ended, process samples and send for training
                # 如果环境结束，处理样本并发送训练
                if done:
                    # Add terminal reward to last sample
                    # 添加最后一帧的终止奖励到最后一个样本
                    if len(collector) > 0:
                        collector[-1].reward = collector[-1].reward + final_reward

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        monitor_data = {
                            "reward": total_reward + final_reward[-1],
                            "episode_cnt": self.episode_cnt,
                            "step_reward": round(self.agent.step_reward_sum, 4),
                            "treasure_reward": round(self.agent.treasure_reward_sum, 4),
                            "survive_reward": round(self.agent.survive_reward_sum, 4),
                            "visited_norm": round(self.agent.preprocessor.visited_norm, 4),
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if len(collector) > 0:
                        collector = sample_process(collector)
                        yield collector
                    break

                # State update
                # 状态更新
                obs_data = _obs_data
                remain_info = _remain_info
                obs = _obs
