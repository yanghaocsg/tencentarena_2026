# PPO v6 设计与优化路线 —— 规则 Bot + 行为克隆（BC）预训练 → PPO 微调

> 项目：腾讯开悟「峡谷追猎」gorge_chase v18.1.7
> 目标：单英雄在 128×128 地图躲避怪物、吃宝箱，活满 1000 步通关
> 版本：v6（在 v4 基线之上新增规则专家 bot 与 BC 预训练；v5 课程学习已证伪、不采用）

---

## 1. 为什么 v6 改走"规则 bot + BC"

v1→v5 的 50 局评估均值：

| 指标 | v1 | v3 | v4（基线） | v5（课程） |
| :-- | --: | --: | --: | --: |
| 总分 | 927.99 | 849.23 | **1004.46** | 882.49 |
| 终止步数 | 342.66 | 320.82 | **365.64** | 333.66 |
| 宝箱数 | 4.14 | 3.68 | **4.56** | 3.82 |
| 技能次数 | 3.78 | 3.64 | 3.88 | 3.66 |
| buff 次数 | 0.12 | 0.16 | **0.22** | 0.10 |

两条路线的结论：

1. **纯逐帧奖励 shaping（v4）边际收益小**：方向正确但两轮只换来约 +23 步，平均仍死在 365 步（双怪包夹），无 WIN、方差极大、几乎不捡 buff。
2. **课程学习（v5）被证伪**：按局数四阶段硬切换，全面退步到 882.49；延长训练也不能救——8h 已在完整环境横盘 300–500 步约 4h 无上升趋势，根因是局部最优 + 熵塌缩 + RND 截断 0.1。

v5 暴露出的本质问题是**冷启动**：策略在掌握基础走位前就被投入完整双怪 + 加速环境，早期高死亡率导致"风筝、捡 buff、穿梭落点、后期生存"的正样本极少，且从零探索容易掉进局部最优。

因此 v6 换思路：**先写一个具备"风筝 / 利用障碍 / 穿梭 / 捡 buff / 探索"能力的规则专家 bot，用它自跑产生示范数据做行为克隆（BC），把策略"热"到一个合理的行为先验，再用 PPO 在其基础上微调。** 这样一次性解决冷启动、落点、方差与 buff 问题，且不需要改动环境与 learner。

---

## 2. 规则专家 bot（rule_bot.py）

### 2.1 核心：双 BFS 可达场 + 安全裕度

bot 读取 Preprocessor 已解析的状态（不重新解析），每步计算两张可达场：

- **英雄可达场**：从英雄 BFS（半径 `HERO_RADIUS × hero_speed`），距离按英雄速度换算成到达时间（`格数 / hero_speed`），并带 parent 用于恢复路径。
- **怪物可达场**：从每只视野内怪物 BFS（半径 `MONSTER_RADIUS`），距离按怪物速度换算（`格数 / monster_speed`）。

每个格子的**安全裕度**：

```
margin(cell) = monster_arrival(cell) - hero_arrival(cell)
```

- 开阔格子：英雄与怪物路径相近，margin 小；
- **墙后格子**：怪物必须绕墙、英雄能快速到达，margin 大。

每步在英雄可达格中选 margin 最大的目标，沿 parent 回溯第一步动作——**天然利用障碍切断追击**，而不是在开阔地里和怪物硬拼速度。

### 2.2 方向安全深度（防切角）

8 方向离散追逐有个关键几何：怪物斜向移动的"效率"是英雄单轴移动的 √2 倍——英雄纯横向（切线）时，怪物用斜向每步同时完成"横向跟随 + 纵向接近"，距离持续缩小。

`_dir_safe_depths` 沿每条射线计算英雄能保持裕度 ≥ 阈值的最大距离（英雄到第 i 格时间为 `i/hero_speed`）：能长期保持裕度的方向（沿走廊）得高分，裕度快速衰减的横切方向（怪物切角）被压分。

### 2.3 穿梭：danger / trapped 双触发

`_maybe_flash` 不再只在危急时触发，而是：

- 怪物加速期：`danger = 怪物距离 ≤ 4` 或 `trapped = 走位裕度 ≤ 0`；
- 同速期：`danger = 距离 ≤ 2` 或 `trapped = 走位裕度 ≤ 1`。

落点要求：开阔度 `openness ≥ 0.20`，封闭时用 `_has_escape`（落点至少 1 个可走相邻格）兜底；穿梭后怪物距离必须显著提升（加速期 +3 / 同速 +2）才值得用掉 100 步 CD。这把穿梭从"几乎不用"变成"在真正被困/被围时穿越缺口、重置位置"的战略工具。

### 2.4 buff 周期续航

- 读取英雄自身 `speed_up`（加速中）与 buff organ 固定位置 `buff_pos`（即使冷却中仍存在、原位刷新）。
- **buff 可用且英雄能先到**（buff_safe）：第一方向 +2.0、缩短路径每格 +0.8，强引导去吃。
- **buff 冷却中**：只在 `buff_cooldown ≤ 当前到 buff 路径 + 10`（即将刷新、该返回）时才引导，加速期英雄可自由逃跑/探索，不提前回防。
- **buff 锚定**：离 buff 超过 40 格时，对继续远离的候选格每格惩罚 0.9，避免跑进远处边缘死路。

### 2.5 探索与边缘

- 未发现 buff 且安全时，朝未访问格（`visit` 掩码）引导，以尽快发现 buff / 宝箱。
- 边缘惩罚累加所有近边缘：角落（两边都近）比沿单一边缘受罚更重，引导英雄在进角落前转向。

---

## 3. BC 工作流（自包含于 aisrv）

BC 阶段完全在 aisrv worker 内完成，不依赖 learner 改动：

1. `workflow()` 在正常 PPO 循环前，若 `Config.BC_ENABLED` 则调用 `run_bc_pretrain`。
2. 每个 worker 跑 `BC_DEMO_EPISODES = 12` 局规则 bot：每步记录 `(feature, legal_action, expert_action)`，并用专家动作与环境交互（设 `agent.last_action = expert_action` 对齐下一帧特征）。
3. 一局结束后用 `BC_EPOCHS = 3` 轮 `algorithm.bc_learn` 更新**该 worker 自己的模型**：
   - mini-batch 256；masked `log_softmax` + NLL 交叉熵（只在合法动作上归一化）；梯度裁剪 0.5。
4. 12 局完成后 `agent.save_model()` 保存 BC 权重，随后进入正常 `EpisodeRunner`。
5. PPO 阶段每局 `load_model(id="latest")` 加载 BC 权重，正常 PPO 微调。

> BC 阶段不 `load_model`（首次训练无 checkpoint），从随机模型开始逐局累积；4 个 worker 各自独立做 BC，最终策略都接近规则 bot。

---

## 4. 改动文件

| 文件 | 类型 | 内容 |
| :-- | :-- | :-- |
| `agent_ppo/rule_bot.py` | 新增 | 规则专家 bot：双 BFS 安全裕度、方向安全深度、danger/trapped 穿梭、buff 续航/锚定、探索、边缘惩罚 |
| `agent_ppo/algorithm/algorithm.py` | 修改 | 新增 `bc_learn(expert_frames)`（masked log_softmax + NLL、mini-batch、梯度裁剪）；mask 重构为 `_apply_mask` / `_masked_softmax` 复用 |
| `agent_ppo/conf/conf.py` | 修改 | 新增 `BC_ENABLED`、`BC_DEMO_EPISODES=12`、`BC_EPOCHS=3`、`BC_BATCH_SIZE=256`、`BC_LEARNING_RATE=3e-4` |
| `agent_ppo/workflow/train_workflow.py` | 修改 | `workflow()` 加 BC 阶段调用；新增 `run_bc_pretrain`；`EpisodeRunner` 逻辑保留 |
| `test_rule_bot_mock.py` | 新增（本地测试） | 纯 Python mock：随机障碍/迷宫、greedy 与 BFS 两种怪物、buff 原位刷新，可本地复现 bot 行为 |
| `test_walls.py`、`debug_wall.py`、`debug_score.py`、`diagnose.py` | 新增（本地测试） | 走廊迷宫与评分诊断脚本 |

**不改动**：model / agent / 环境配置 toml / 官方 preprocessor（bot 直接读取其已有输出）。

---

## 5. 本地 mock 验证

本地无 kaiwudrl 与真实地图，用自写 mock 验证逻辑（怪物 BFS 为最坏情况）：

| 配置（各 15 seed） | 结果 |
| :-- | :-- |
| 开阔无 buff，greedy | 平均 380 步，min 125 / max 507（双峰） |
| 随机障碍 8% + buff，**BFS 完美怪物** | 平均 **480 步**，中位 507，min 321 / max 517，mean 穿梭 1.7 |
| 交错走廊迷宫，greedy（seed0） | 413 步、1 穿梭，能利用墙拉开 |

关键结论：

- bot 会吃第一个 buff、加速期能把怪物距离拉开到 40+ 格（远优于早期版本的 7–9 格），会利用障碍、会在被困时穿梭。
- 开阔无 buff 固定在 500–507 步死，符合**数学结论**：怪物 500 步加速后，开阔走位 + 有限穿梭（CD 100）无法无限风筝。
- 第二次 buff 续航（在 200 步周期内折返、持续覆盖到 500 步后）是高级动态博弈，纯反应式规则仍不稳定；这部分预期由 BC + PPO 微调在真实环境中继续优化。

> 所有正式文件 `py_compile` 通过；本地无法跑真实训练，真实效果以平台运行为准。

---

## 6. 提交后监控判据

1. **BC 阶段日志**：应出现 `[BC] pretrain start`，逐局 `[BC] demo k/12 ... loss: ...`，loss 应随局数下降；最后 `[BC] pretrain finished, model saved`。
2. **快信号（BC 刚切 PPO 时）**：
   - 开局 `finished_steps` 应明显高于 v4 的早期随机水平，速死局（<100 步）显著减少；
   - 穿梭集中在被困/包夹时刻而非 CD 好就用；
   - buff 拾取次数明显抬升。
3. **PPO 微调阶段**：平均存活应越过 v4 的 365 向 500+ 推进，并开始出现 WIN；50 局评估总分应稳定超过 1004.46。
4. 核心看**终止步数 / WIN 数 / 穿梭时机**，而非仅总分。

若 BC 阶段在平台跑不通（save/load 机制差异），备选：把示范帧标记后送 learner 做 BC loss（需改 algorithm 与 SampleData 标记）。

---

## 7. 风险与后续方向

- **真实怪物寻路未最终确认**：官方 feature_utils 评估威胁全用欧氏直线距离（暗示贪心），但真实评估平均 365 步又说明怪物不完全是被墙卡死的纯贪心。v6 以 BFS 为最坏情况设计，若真实怪物偏贪心，bot 与 BC 策略会更从容。
- **4 worker 下 BC 权重的保存/加载**需以平台首次运行日志验证。
- 后续可继续：① 加强规则 bot 的"以 buff 为据点做轨道运动"；② PPO 中对 BC 先验加较低权重的 KL/BC 正则，防止微调早期偏离；③ 若 BC + PPO 稳定，再评估多头价值与 PPG 等结构升级。
