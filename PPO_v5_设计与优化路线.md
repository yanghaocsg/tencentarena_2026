# PPO v5 设计与优化路线 —— 课程学习（Curriculum Learning）

> 项目：腾讯开悟「峡谷追猎」gorge_chase v18.1.7
> 目标：单英雄在 128×128 地图躲避怪物、吃宝箱，活满 1000 步通关
> 版本：v5（在 v4 基础上新增课程学习；algorithm.py 沿用 v4，含 advantage 标准化）

---

## 1. 为什么 v5 要做课程学习

v1→v4 的 50 局评估均值：

| 指标 | v1 | v3 | v4 |
| :-- | --: | --: | --: |
| 总分 | 927.99 | 849.23 | **1004.46** |
| 终止步数 | 342.66 | 320.82 | **365.64** |
| 宝箱数 | 4.14 | 3.68 | **4.56** |
| 技能次数 | 3.78 | 3.64 | 3.88 |
| buff 次数 | 0.12 | 0.16 | **0.22** |

v4 的双怪几何 / 落点 / 包夹 / buff 奖励方向正确、各指标连续改善，但**纯逐帧奖励 shaping 的边际收益已经很小**（两轮只换来约 +23 步），且三个结构性问题仍在：

1. 平均仍只活 ~365 步，死在第二只怪（300 步出现）后的双怪包夹，活不到 500 加速；无 WIN（最好 675 步）。
2. 方差极大（29 步速死 vs 675 步）。
3. buff 几乎不捡，技能仍偏"CD 好就用"。

这些是**探索/样本结构**问题：智能体在掌握基础走位前就被投入完整双怪+加速环境，早期高死亡率导致"双怪落点、捡 buff、后期生存"的正样本太少。课程学习通过**由易到难地安排环境配置**，让每个阶段集中产生对应技能的正样本。

---

## 2. v5 核心设计：四阶段课程

进度信号采用 **EpisodeRunner 累计完成的局数 `episode_cnt`**（每个 aisrv worker 独立计数，环境侧直接可得，不依赖 learner 的 global_step）。每局 `env.reset` 前按当前局数在基础配置上覆盖环境参数。

| 阶段 | 局数（每 worker） | monster_interval | monster_speedup | treasure | buff_cd | 训练重点 |
| :-- | :-- | --: | --: | --: | --: | :-- |
| **0_single** | 0–599 | 2000（第二怪不出现） | 2000（不加速） | 6 | 200 | 单怪风筝、基础走位与收集 |
| **1_dual_nospeed** | 600–1799 | 150（双怪早出现） | 2000（不加速） | 10 | **50（高频刷 buff）** | 双怪风筝、穿梭落点、学会捡 buff |
| **2_dual_speed** | 1800–2999 | 300 | 500（恢复加速） | 10 | 200 | 500 步加速后的后期生存 |
| **3_full** | ≥3000 | 300 | 500 | 10 | 200 | 完整比赛环境精调（= 基础 toml） |

设计要点：

- **阶段 0**：`monster_interval=2000`（>max_step=1000）等同单怪，`speedup=2000` 不加速，宝箱减到 6，先把走位/收集学扎实，显著降低开局速死。
- **阶段 1**：第二怪提前到 150 步出现，拉长双怪练习窗口；怪物仍不加速，先在低压力下练双怪风筝与穿梭落点；`buff_cooldown=50` 让 buff 高频刷新，配合 v4 的 buff 拾取/靠近奖励，先学会捡 buff。
- **阶段 2**：恢复完整怪物节奏（300 出现、500 加速），集中学习后期加速压力下的生存。
- **阶段 3**：无任何覆盖，完全使用 `train_env_conf.toml`（比赛真实配置）精调。
- 所有覆盖取值均在官方合法范围内（`talent_cooldown` 下限 100，故未缩短技能 CD，靠双怪早出现增加练习；`buff_cooldown` 范围 1–500，取 50 合法）。

> 阶段边界（600/1800/3000）集中在 `curriculum.py` 的 `STAGES` 常量，可按实际训练时长/收敛速度调整。

---

## 3. 实现与改动文件

### 3.1 新增 `agent_ppo/conf/curriculum.py`

- `STAGES`：各阶段局数上界与配置覆盖表；`STAGE_NAMES`：阶段名。
- `get_stage_index / get_stage_name`：由局数判定阶段。
- `build_curriculum_conf(base_conf, episode_cnt)`：用 `copy.deepcopy` 复制基础配置后覆盖当前阶段参数；**自动适配扁平 dict 与嵌套 `{"env_conf": {...}}` 两种 usr_conf 结构**；最终阶段原样返回基础配置。

### 3.2 修改 `agent_ppo/workflow/train_workflow.py`

- import `build_curriculum_conf, get_stage_name`。
- 每局 `env.reset` 前：
  ```python
  stage_name = get_stage_name(self.episode_cnt)
  cur_conf = build_curriculum_conf(self.usr_conf, self.episode_cnt)
  env_obs = self.env.reset(cur_conf)
  ```
- 开局日志打印当前阶段与本局配置：`Episode N stage 1_dual_nospeed start, usr_conf is ...`，便于在训练日志中确认课程推进。

### 3.3 修改 `agent_ppo/feature/preprocessor.py`（相位对齐）

课程会改变 `monster_interval`（150/2000），原先硬编码的 300/500 相位阈值会与实际威胁错配，故：

- 新增 `_get_threat_timings()`：从 `self.env_info` 读取实际 `monster_interval / monster_speedup`，缺失或非法（−1/None）时回退比赛默认 300/500。
- `_calc_phase_reward` 的两段相位阈值改用 `_get_threat_timings()`，使各阶段收集/生存权重与实际威胁同步。
- `_calc_flash_landing_reward` 的 `is_dual_phase` 判定由 `frame_no >= 300` 改为 `frame_no >= 实际第二怪出现步`，使阶段 1（150 步双怪）能正确触发双怪落点奖励。

### 3.4 不改动

- `algorithm.py`：沿用 v4（含 v2 加入、判定正确的 **advantage batch 标准化**），本轮无新增改动。
- model / agent / 其余 conf 均不动。

---

## 4. 验证情况

- `py_compile`：`curriculum.py`、`train_workflow.py`、`preprocessor.py` 全部通过。
- 课程逻辑功能性自测：阶段边界（599/600、1799/1800、2999/3000）切换正确；各阶段覆盖值正确；扁平与嵌套结构均生效；`deepcopy` 保证基础配置不被修改。
- 本地无 kaiwudrl/模拟器，无法本地跑训练；需提交集群验证。

---

## 5. 提交后监控判据

训练时长建议与 v4 相当（约 10h），按课程推进观察：

1. **训练日志**：开局日志的 `stage` 字段应随局数依次出现 `0_single → 1_dual_nospeed → 2_dual_speed → 3_full`，确认课程在推进。
2. **快信号**：
   - 阶段 0/1 前期 `finished_steps` 应明显抬升、速死局（<100 步）占比下降；
   - 阶段 1 buff 获得次数应明显抬升（目标 >0.5，最终接近 1+）；
   - 进入阶段 2/3 后，平均存活应越过 365 向 500+ 推进，并开始出现 WIN。
3. **总分**：50 局评估总分应稳定超过 v4 的 1004.46，核心看终止步数与 WIN 数，而非仅总分。

若课程在某一阶段长期卡住（例如进入加速阶段后仍集体死亡），可：① 调整该阶段局数上界、延长练习；② 微调该阶段环境参数（如进一步延后 speedup）。

---

## 6. 后续方向

- 若课程学习后仍无法稳定通关，下一步上 **规则 bot + 行为克隆（BC）**：编写会"双怪风筝（绕障碍分割两怪）+ 留技能到包夹 + 捡 buff + 开局不送"的规则脚本，自跑生成数据做 BC 预训练，再用 PPO 微调，可同时解决落点、方差、buff 问题。
- 进入 BC 前需确认开悟支持加载 BC 预训练权重后再做 PPO 微调（模型 load 即可）。
