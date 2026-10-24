#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Rule-based expert bot for Gorge Chase (used for behavioral cloning pretrain).
峡谷追猎规则专家机器人（用于行为克隆 BC 预训练）。

The bot reads the state already parsed by Preprocessor and outputs one action
in [0, 15] that is guaranteed to be legal.
规则 bot 读取 Preprocessor 已解析的状态，输出一个 0-15 且合法的动作。

Action convention (same as env, z axis points down, clockwise):
0 right(1,0)  1 lower-right(1,1)  2 down(0,1)   3 lower-left(-1,1)
4 left(-1,0)  5 upper-left(-1,-1) 6 up(0,-1)    7 upper-right(1,-1)
8-15 are the spatial-flash versions of the same eight directions.

Strategy / 策略:
- Compute a hero reachable field (BFS from hero) and a monster reachable field
  (BFS from each monster). The safety margin of a cell is
  monster_dist - hero_dist. Cells behind walls have a large margin because the
  monster must detour while the hero can reach them quickly.
  计算英雄可达场（英雄 BFS）与怪物可达场（怪物 BFS），格子安全裕度 = 怪物距离
  - 英雄距离；墙后格子因怪物绕路、英雄可达而裕度大。
- Each step the bot moves toward the reachable cell with the largest margin,
  which naturally uses obstacles to break pursuit.
  每步朝裕度最大的可达格子移动，天然利用障碍切断追击。
- Flash is reserved for the post-speedup phase and true emergencies.
  穿梭保留给加速后阶段与真正危急时刻。
"""

import math
from collections import deque


# Move deltas for action 0-7, index aligned with the action id
# 动作 0-7 的位移增量，下标与动作 id 对齐
MOVE_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

# Flash travel distance per direction: cardinal 10 cells, diagonal 8 cells
# 各方向穿梭距离：直线 10 格，斜向 8 格
FLASH_LEN = [10, 8, 10, 8, 10, 8, 10, 8]

# Radius for the hero reachable field (search for a safe target)
# 英雄可达场半径（搜索安全目标）
HERO_RADIUS = 18

# Radius for the monster reachable field (must cover detours around walls)
# 怪物可达场半径（需覆盖绕墙迂回）
MONSTER_RADIUS = 30


def hypot(dx, dz):
    return math.sqrt(dx * dx + dz * dz)


class RuleBot:
    """Deterministic expert policy based on reachable-field safety margins."""

    def __init__(self, map_size=128):
        self.map_size = map_size
        self.recent_positions = []

    def reset(self):
        self.recent_positions = []

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def act(self, fm, legal_action):
        x, z = fm.cur_pos
        grid = fm.map_data[0]
        # Visited mask (1 = visited). When safe and no buff is in view, the hero
        # should explore unvisited cells to discover buffs / treasures.
        # 已访问掩码（1=已访问）。安全且未发现 buff 时，英雄应探索未访问区域以发现 buff/宝箱。
        visit = fm.map_data[1] if len(fm.map_data) > 1 else None
        frame_no = getattr(fm, "frame_no", 0)

        # Monsters speed up to 2 cells/step after default step 500
        # 怪物在默认 500 步后加速到 2 格/步
        monster_speed = 2 if frame_no >= 500 else 1
        # Hero moves 2 cells/step while the speed-up buff is active
        # 英雄拾取 buff 后也以 2 格/步移动
        hero_speed = 2 if getattr(fm, "speed_up", False) else 1

        monsters = self._collect_monsters(fm)
        treasures = self._collect_treasures(fm)
        buff = self._collect_buff(fm)
        # Buff availability and respawn cooldown (0 when available)
        # buff 当前是否可用、刷新冷却（可用时为 0）
        buff_available = bool(getattr(fm, "buff_status", False))
        buff_cooldown = int(getattr(fm, "buff_cooldown", 0))

        # Hero reachable field (BFS from hero, with parent for path recovery).
        # Distance is converted to hero arrival time (steps) using hero_speed.
        # 英雄可达场（英雄 BFS，带 parent 用于恢复路径）；距离按英雄速度换算成到达时间。
        hero_cells, hero_parent = self._bfs_from(
            grid, x, z, HERO_RADIUS * hero_speed, speed=1
        )
        hero_dist = {k: v / float(hero_speed) for k, v in hero_cells.items()}
        # Monster reachable field (BFS from each monster)
        # 怪物可达场（每只怪物 BFS）
        monster_dist = self._bfs_monsters(grid, monsters, MONSTER_RADIUS, monster_speed)

        # Safe depth in each direction: how far along the ray the hero keeps a
        # positive safety margin (monster cannot cut the corner to catch up).
        # 各方向安全深度：沿射线英雄能保持正安全裕度的距离（怪物无法切角追上）。

        def mdist(px, pz):
            d = monster_dist.get((px, pz), None)
            return MONSTER_RADIUS + 4.0 if d is None else d

        dir_safe = self._dir_safe_depths(grid, x, z, mdist, hero_speed)

        # Buff path field: when the hero can reach the buff before the monster,
        # actively steer toward it (the buff is required after the speed-up).
        # buff 路径场：当英雄能比怪物先到 buff 时，主动朝 buff 移动（加速后必需）。
        buff_dist = None
        buff_safe = False
        buff_first_dir = None
        if buff is not None:
            buff_cells, buff_parent = self._bfs_from(
                grid, buff[0], buff[1], HERO_RADIUS * hero_speed, speed=1
            )
            buff_dist = {k: v / float(hero_speed) for k, v in buff_cells.items()}
            h_to_buff = buff_dist.get((x, z), None)
            if h_to_buff is not None and mdist(buff[0], buff[1]) > h_to_buff + 1.0:
                buff_safe = True
                # First cell along the shortest path toward the buff, and its
                # action id; used to steer strongly while still safe.
                first_cell = buff_parent.get((x, z), None)
                if first_cell is not None:
                    for d in range(8):
                        bdx, bdz = MOVE_DIRS[d]
                        if (x + bdx, z + bdz) == first_cell:
                            buff_first_dir = d
                            break

        cur_margin = mdist(x, z) - 0.0

        # 1) Best walk target and its safety margin
        # 1) 走位最佳目标及其安全裕度
        target_cell, walk_margin = self._select_cell(
            x, z, grid, hero_dist, hero_parent, mdist,
            treasures, buff, monster_speed, dir_safe,
            buff_dist, buff_safe, buff_first_dir, visit,
            buff_available, buff_cooldown
        )

        # 2) Flash decision: when in danger or walking cannot keep safe
        # 2) 穿梭决策：紧急或走位无法保持安全时
        if self._skill_usable(legal_action, fm):
            flash_action = self._maybe_flash(
                x, z, grid, mdist, monster_speed, monsters, walk_margin
            )
            if flash_action is not None:
                self._record(x, z)
                return flash_action

        # 3) Recover the first move along the hero BFS path to target_cell
        # 3) 沿英雄 BFS 路径恢复到目标的第一步
        move = self._first_step(
            x, z, target_cell, hero_parent, legal_action, monster_speed
        )
        self._record(x, z)
        return move

    # ------------------------------------------------------------------
    # State collection
    # ------------------------------------------------------------------
    def _collect_monsters(self, fm):
        out = []
        for m in getattr(fm, "all_monsters", []):
            mx, mz = m["pos"]["x"], m["pos"]["z"]
            visible = m.get("is_in_view", 0) == 1 and mx >= 0 and mz >= 0
            if visible:
                out.append((mx, mz))
        return out

    def _collect_treasures(self, fm):
        out = []
        for t in getattr(fm, "treasure_list", []):
            if t.get("status", 1) == 1:
                out.append((t["pos"]["x"], t["pos"]["z"]))
        return out

    def _collect_buff(self, fm):
        # buff_pos is the fixed location of the buff organ, exposed once the
        # hero has discovered it (buff_found). Availability is read separately
        # via buff_status; the cooldown via buff_cooldown.
        # 英雄一旦发现 buff（buff_found），buff_pos 即固定位置；是否可用看
        # buff_status，刷新剩余看 buff_cooldown。
        if getattr(fm, "buff_found", False):
            bp = getattr(fm, "buff_pos", None)
            if bp is not None and len(bp) == 2 and bp[0] >= 0 and bp[1] >= 0:
                return (bp[0], bp[1])
        return None

    def _skill_usable(self, legal_action, fm):
        if sum(legal_action[8:16]) > 0:
            return True
        return bool(getattr(fm, "skill_usable", False))

    # ------------------------------------------------------------------
    # BFS fields
    # ------------------------------------------------------------------
    def _bfs_from(self, grid, sx, sz, radius, speed=1):
        """BFS from (sx,sz), returning distance (in cells) and parent map."""
        dist = {(sx, sz): 0}
        parent = {(sx, sz): None}
        q = deque([(sx, sz, 0)])
        while q:
            x, z, d = q.popleft()
            if d >= radius:
                continue
            nd = d + 1
            for dx, dz in MOVE_DIRS:
                nx, nz = x + dx, z + dz
                if not self._passable(grid, nx, nz):
                    continue
                if (nx, nz) not in dist:
                    dist[(nx, nz)] = nd
                    parent[(nx, nz)] = (x, z)
                    q.append((nx, nz, nd))
        return dist, parent

    def _bfs_monsters(self, grid, monsters, radius, speed):
        """BFS from every monster; distance in monster steps (cells/speed)."""
        cells = {}
        q = deque()
        for mx, mz in monsters:
            q.append((mx, mz, 0))
            cells[(mx, mz)] = 0
        while q:
            x, z, d = q.popleft()
            if d >= radius:
                continue
            nd = d + 1
            for dx, dz in MOVE_DIRS:
                nx, nz = x + dx, z + dz
                if not self._passable(grid, nx, nz):
                    continue
                if (nx, nz) not in cells:
                    cells[(nx, nz)] = nd
                    q.append((nx, nz, nd))
        # Convert cell distance to monster arrival time (steps)
        # 路径格数换算成怪物到达时间（步）
        return {k: v / float(speed) for k, v in cells.items()}

    # ------------------------------------------------------------------
    # Target cell selection
    # ------------------------------------------------------------------
    def _select_cell(self, x, z, grid, hero_dist, hero_parent, mdist,
                     treasures, buff, speed, dir_safe, buff_dist, buff_safe,
                     buff_first_dir, visit, buff_available, buff_cooldown):
        safe_time = 3.0 if speed < 2 else 7.0
        best_cell, best_score, best_margin = (x, z), -1e9, -1e9
        cur_bdist = buff_dist.get((x, z), None) if buff_dist is not None else None

        for cell, hd in hero_dist.items():
            if hd < 1:
                continue
            md = mdist(cell[0], cell[1])
            # Safety margin: how much earlier than the monster can the hero be
            # 安全裕度：英雄相比怪物能早到多少
            margin = md - hd
            score = margin * 1.4
            # Prefer staying at a comfortable distance rather than hugging the
            # monster; marginal extra monster distance is still good.
            score += min(md, 12.0) * 0.35
            # First-step direction: reward directions that keep a safe margin
            # over a long distance (along a corridor), penalize lateral moves
            # whose margin collapses (monster cuts the corner).
            # 第一步方向：奖励能长期保持安全裕度的方向（沿走廊），惩罚裕度快速
            # 衰减的横向移动（怪物切角）。
            first_d = self._first_step_dir(x, z, cell, hero_parent)
            if first_d is not None:
                score += (dir_safe[first_d] / 30.0) * 2.4

            # Buff path guidance.
            # buff 路径引导：
            # (a) available and reachable before the monster: strong steer to eat.
            # (a) 可用且英雄能先到：强引导去吃。
            # (b) on cooldown: while safe, return toward the fixed buff so the
            # hero is there when it respawns (stronger if currently too far).
            # (b) 冷却中：安全时返回固定 buff，确保刷新时在附近；离得越远引导越强。
            if buff_available and buff_safe and margin >= 1.0:
                if first_d == buff_first_dir:
                    score += 2.0
                if cur_bdist is not None:
                    cell_bdist = buff_dist.get(cell, None)
                    if cell_bdist is not None and cell_bdist < cur_bdist:
                        score += (cur_bdist - cell_bdist) * 0.8
            elif (not buff_available) and buff_dist is not None and margin >= 2.0:
                # Only start returning when the buff is about to respawn:
                # cooldown <= current path-to-buff + buffer. Until then the
                # hero is free to flee / explore during the speed-up.
                # 仅当 buff 即将刷新（cooldown <= 当前到 buff 路径 + 缓冲）时才返回，
                # 加速期英雄可自由逃跑/探索。
                need_return = (cur_bdist is not None
                               and buff_cooldown <= cur_bdist + 10)
                too_far = cur_bdist is not None and cur_bdist > buff_cooldown
                if need_return:
                    if first_d == buff_first_dir:
                        score += 1.6 if too_far else 0.9
                    cell_bdist = buff_dist.get(cell, None)
                    if cur_bdist is not None and cell_bdist is not None \
                            and cell_bdist < cur_bdist:
                        score += (cur_bdist - cell_bdist) * (0.7 if too_far else 0.4)
                # Buff anchoring: once far from the buff, penalize moving even
                # farther so the hero does not run into a remote edge pocket.
                # buff 锚定：离 buff 已远时，惩罚继续远离，避免跑进远处边缘死路。
                if cur_bdist is not None and cur_bdist > 40:
                    cell_bdist = buff_dist.get(cell, None)
                    if cell_bdist is not None and cell_bdist > cur_bdist:
                        score -= (cell_bdist - cur_bdist) * 0.9
            # Treasure guidance only when safe
            # 安全时引导宝箱
            if treasures and md >= safe_time:
                for tx, tz in treasures:
                    if hypot(tx - cell[0], tz - cell[1]) < 2:
                        score += 0.7
                        break

            # Exploration: when safe and no buff is visible, steer toward
            # unvisited cells to discover buffs / treasures.
            # 探索引导：安全且未发现 buff 时，朝未访问格移动以发现 buff/宝箱。
            if buff is None and margin >= 2.0 and visit is not None:
                vx, vz = cell
                if (0 <= vx < visit.shape[0] and 0 <= vz < visit.shape[1]
                        and visit[vx, vz] == 0):
                    score += 1.3

            # Anti-stuck
            # 防卡死
            if cell in self.recent_positions:
                score -= 1.0
            # Edge
            # 边缘
            score -= self._edge_penalty(cell[0], cell[1]) * 2.5

            if score > best_score:
                best_score, best_cell, best_margin = score, cell, margin

        return best_cell, best_margin

    def _first_step_dir(self, x, z, cell, parent):
        """Return the action id of the first step from hero toward cell."""
        cur = cell
        guard = 0
        while parent.get(cur, None) is not None and parent[cur] != (x, z):
            cur = parent[cur]
            guard += 1
            if guard > 60:
                return None
        for d in range(8):
            dx, dz = MOVE_DIRS[d]
            if (x + dx, z + dz) == cur:
                return d
        return None

    # ------------------------------------------------------------------
    # First step toward target cell
    # ------------------------------------------------------------------
    def _first_step(self, x, z, target_cell, parent, legal_action, speed):
        # Walk the parent chain from target back to the cell adjacent to hero
        # 从 target 沿 parent 回溯到英雄相邻格
        cur = target_cell
        guard = 0
        while parent.get(cur, None) is not None and parent[cur] != (x, z):
            cur = parent[cur]
            guard += 1
            if guard > 60:
                break

        # cur is now the first step cell; find its action id
        # cur 现在是第一步格子，找对应动作 id
        for d in range(8):
            dx, dz = MOVE_DIRS[d]
            if (x + dx, z + dz) == cur and legal_action[d] == 1:
                return d

        # Fallback: any legal move, else 0
        # 兜底：任意合法动作，否则 0
        for d in range(8):
            if legal_action[d] == 1:
                return d
        return 0

    # ------------------------------------------------------------------
    # Flash decision
    # ------------------------------------------------------------------
    def _maybe_flash(self, x, z, grid, mdist, speed, monsters, walk_margin):
        cur_md = mdist(x, z)
        if speed >= 2:
            danger = cur_md <= 4.0
            trapped = walk_margin <= 0.0
        else:
            danger = cur_md <= 2.0
            trapped = walk_margin <= 1.0
        if not danger and not trapped:
            return None

        best_d, best_md = None, -1e9
        for d in range(8):
            lx, lz, moved = self._flash_landing(x, z, grid, d)
            if not moved:
                continue
            # The landing must not be a closed pocket; require either an open
            # area or enough reachable walkable cells.
            # 落点不能是封闭口袋：要求开阔或有足够可走格。
            if self._openness(lx, lz, grid) < 0.20:
                # fallback: at least one further walkable cell from landing
                if not self._has_escape(grid, lx, lz):
                    continue
            md = mdist(lx, lz)
            if md > best_md:
                best_md, best_d = md, d

        if best_d is None:
            return None
        # Flash must improve the monster distance enough to be worth the CD.
        # 穿梭后怪物距离必须显著提升，才值得用掉冷却。
        gain = 3.0 if speed >= 2 else 2.0
        if best_md <= cur_md + gain:
            return None
        return 8 + best_d

    def _has_escape(self, grid, x, z):
        for dx, dz in MOVE_DIRS:
            if self._passable(grid, x + dx, z + dz):
                return True
        return False

    def _flash_landing(self, x, z, grid, d):
        dx, dz = MOVE_DIRS[d]
        length = FLASH_LEN[d]
        last = (x, z)
        for i in range(1, length + 1):
            nx, nz = x + dx * i, z + dz * i
            if self._passable(grid, nx, nz):
                last = (nx, nz)
            else:
                break
        return last[0], last[1], last != (x, z)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _passable(self, grid, x, z):
        if x < 0 or z < 0 or x >= self.map_size or z >= self.map_size:
            return False
        return grid[x, z] == 1

    def _openness(self, x, z, grid):
        """Fraction of the eight neighboring cells that are passable.

        周围 8 格可走的比例。
        """
        cnt = 0
        for dx, dz in MOVE_DIRS:
            if self._passable(grid, x + dx, z + dz):
                cnt += 1
        return cnt / 8.0

    def _dir_safe_depths(self, grid, x, z, mdist, hero_speed=1, cap=30, margin_thresh=2.0):
        """For each direction, the distance along the ray where the hero keeps a
        safety margin >= margin_thresh (monster cannot cut the corner).

        Hero arrival time at the i-th cell is i/hero_speed, so the margin is
        computed against that time.
        每个方向：沿射线英雄保持安全裕度 >= 阈值的最大距离；英雄到第 i 格时间为
        i/hero_speed，裕度据此计算。
        """
        out = []
        for dx, dz in MOVE_DIRS:
            depth = 0
            for i in range(1, cap + 1):
                nx, nz = x + dx * i, z + dz * i
                if not self._passable(grid, nx, nz):
                    break
                margin = mdist(nx, nz) - i / float(hero_speed)
                if margin >= margin_thresh:
                    depth = i
                else:
                    break
            out.append(depth)
        return out

    def _edge_penalty(self, x, z):
        # Sum the penalty over every near edge so a corner (two near edges) is
        # penalized more heavily than running along a single edge.
        # 累加所有近边缘的惩罚，使角落（两边都近）比沿单一边缘受罚更重。
        pen = 0.0
        EDGE = 24
        for d in (x, z, self.map_size - 1 - x, self.map_size - 1 - z):
            if d < EDGE:
                pen += (EDGE - d) / EDGE
        return pen

    def _record(self, x, z):
        self.recent_positions.append((x, z))
        if len(self.recent_positions) > 6:
            self.recent_positions.pop(0)
