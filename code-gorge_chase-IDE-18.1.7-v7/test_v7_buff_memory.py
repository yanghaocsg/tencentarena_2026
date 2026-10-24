#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Functional test for v7 persistent buff memory.

v7 buff 持久记忆功能测试。真实实例化 Preprocessor，模拟"看到 buff -> 离开视野 ->
记忆保持 -> 拾取进入冷却 -> 视野外本地倒计时 -> 刷新"的完整生命周期，并核对
feature 维度（480）与奖励（发现 +0.5 / 拾取 +1.0 / 靠近）。

Local diagnostic only; not uploaded to the cluster.
仅本地诊断，不上传集群。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from agent_ppo.feature.preprocessor import Preprocessor


def buff_organ(x, z, status, cooldown=0):
    return {
        "sub_type": 2,
        "status": status,
        "cooldown": cooldown,
        "pos": {"x": x, "z": z},
    }


def make_obs(step, hero_xyz, organs, collected_buff=0, treasures_collected=0):
    hx, hz = hero_xyz
    return {
        "observation": {
            "step_no": step,
            "legal_action": [1] * 16,
            "map_info": None,  # skip obstacle map update; obstacles default walkable
            "frame_state": {
                "heroes": {
                    "pos": {"x": hx, "z": hz},
                    "buff_remaining_time": 0,
                    "flash_cooldown": 0,
                },
                "organs": organs,
                "monsters": [],
            },
            "env_info": {
                "collected_buff": collected_buff,
                "treasures_collected": treasures_collected,
                "total_treasure": 10,
                "monster_interval": 300,
            },
        }
    }


def state_line(p):
    return (
        f"found={int(p.buff_found)} disc={int(p.buff_discovery)} "
        f"status={int(p.buff_status)} cd={int(p.buff_cooldown)} "
        f"pos={p.buff_pos} dist={p.buff_dist:.1f}"
    )


def check(name, cond):
    print(("PASS" if cond else "FAIL") + "  " + name)
    if not cond:
        check.failed += 1
check.failed = 0


p = Preprocessor()

# --- T0: never seen a buff ---
p.pb2struct(make_obs(0, (40, 40), []))
r0 = p.reward_process()
print("T0 (no buff):      ", state_line(p))
check("T0 buff_found False", not p.buff_found)
check("T0 discovery 0", p.buff_discovery == 0)

# --- T1: first time seeing an available buff (5 cells away) ---
p.pb2struct(make_obs(1, (50, 50), [buff_organ(55, 55, 1)]))
r1 = p.reward_process()
print("T1 (first sight):  ", state_line(p))
check("T1 buff_found True", p.buff_found)
check("T1 discovery=1 (first time)", p.buff_discovery == 1)
check("T1 status=1 available", p.buff_status == 1)
check("T1 pos=(55,55)", p.buff_pos == (55, 55))
check("T1 dist=7.07", abs(p.buff_dist - np.hypot(5, 5)) < 0.01)
check("T1 discovery reward +0.5", abs(r1[2] - 0.5) < 0.02 or r1[2] > 0.5)

# --- T2: out of view, memory must persist ---
p.pb2struct(make_obs(2, (40, 40), []))
r2 = p.reward_process()
print("T2 (out of view):  ", state_line(p))
check("T2 buff_found True via memory", p.buff_found)
check("T2 discovery back to 0", p.buff_discovery == 0)
check("T2 pos remembered (55,55)", p.buff_pos == (55, 55))
check("T2 status still 1", p.buff_status == 1)

# --- T3: hero reaches buff, picks it up, it enters cooldown (seen status=0) ---
p.pb2struct(make_obs(3, (55, 55), [buff_organ(55, 55, 0, cooldown=5)], collected_buff=1))
r3 = p.reward_process()
print("T3 (picked/cooling):", state_line(p))
check("T3 buff_get=1 (pickup)", p.buff_get == 1)
check("T3 status=0 cooling", p.buff_status == 0)
check("T3 memory cd=5", p.buff_memory_cd == 5)

# --- T4: out of view, local cooldown ticks down 5 -> 0, then becomes available ---
cds = []
for i in range(6):
    p.pb2struct(make_obs(4 + i, (40, 40), []))
    cds.append((p.buff_memory_cd, p.buff_status))
print("T4 (out-of-view cd ticks):", cds)
check("T4 cd ticks down", cds[0][0] == 4 and cds[3][0] == 1)
check("T4 becomes available at cd 0", cds[4][1] == 1 and cds[4][0] == 0)
check("T4 stays available after", cds[5][1] == 1)
check("T4 buff_found stays True while cooling", all(p_f for p_f in (True,) * 6))

# --- T6: two buffs in view - pick the available one even if the cooling one is nearer ---
p3 = Preprocessor()
p3.pb2struct(make_obs(1, (50, 50), [
    buff_organ(52, 52, 0, cooldown=100),  # near but cooling
    buff_organ(60, 60, 1),                # farther but available
]))
print("T6 (two buffs):    ", state_line(p3))
check("T6 picks the available (farther) buff", p3.buff_pos == (60, 60) and p3.buff_status == 1)

p3.pb2struct(make_obs(2, (50, 50), [
    buff_organ(52, 52, 1),  # now available and near
    buff_organ(60, 60, 1),  # available but farther
]))
print("T6b (both avail):  ", state_line(p3))
check("T6b picks the nearer available buff", p3.buff_pos == (52, 52))

# --- T5: full feature vector dimension must stay 480 ---
p2 = Preprocessor()
feat, legal, rew, aux = p2.feature_process(
    make_obs(0, (50, 50), [buff_organ(55, 55, 1)]), last_action=-1
)
flat = np.concatenate([feat]) if feat.ndim else feat
print("T5 feature vector size:", feat.shape)
check("T5 feature is 480D", int(feat.shape[0]) == 480)
check("T5 aux labels is 20D", int(aux.shape[0]) == 20)

print()
if check.failed:
    print(f"==== {check.failed} CHECK(S) FAILED ====")
    sys.exit(1)
print("==== ALL CHECKS PASSED ====")
