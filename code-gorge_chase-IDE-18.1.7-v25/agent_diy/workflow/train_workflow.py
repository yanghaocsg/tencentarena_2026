#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""v25: safety-distance-field planner (planning-driven survival).

Core upgrades over reactive BFS (v17/v18):
1. Global stitched map (gmap) with -1 unknown / 0 blocked / 1 passable.
2. Monster memory (last known cell while out of view).
3. Two BFS fields on the SAME map: hero reachable field and monster
   reachable field. Arrival time = cells / speed. Safety margin of a cell
   = monster_arrival - hero_arrival.
4. Only cells with margin >= threshold are "safe"; the safe cells form a
   subgraph. We compute connected components of the safe subgraph and
   strongly prefer the largest component (deep open space with many
   escape routes), avoiding narrow pockets that get cut off.
5. Target guidance (buff / treasure / unexplored) while safe; flash on
   emergencies.
"""
import time
from collections import deque

import numpy as np

from tools.train_env_conf_validate import read_usr_conf

GRID = 128
PASSABLE, BLOCKED, UNKNOWN = 1, 0, -1
DIRS = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
FLASH_LEN = [10, 8, 10, 8, 10, 8, 10, 8]

HERO_RADIUS = 40
MONSTER_RADIUS = 60
SAFE_THRESH = 2.0
EMERGENCY_MARGIN = 1.0


def out(msg):
    print(msg, flush=True)


def get_pos(d):
    if not d:
        return None
    p = d.get("pos")
    if not p:
        return None
    x, z = p.get("x"), p.get("z")
    if x is None or z is None or x < 0 or z < 0:
        return None
    return int(x), int(z)


def parse_view(mi):
    if not mi:
        return None
    if isinstance(mi[0], dict):
        return np.array([x["values"] for x in mi], dtype=np.int32)
    return np.array(mi, dtype=np.int32)


def merge_view(gmap, view, pos):
    vs = view.shape[0]
    half = vs // 2
    px, pz = pos
    l = max(0, px - half)
    r = min(GRID, px + half + 1)
    u = max(0, pz - half)
    d = min(GRID, pz + half + 1)
    vl = l - (px - half)
    vu = u - (pz - half)
    gmap[l:r, u:d] = view[vl:vl + (r - l), vu:vu + (d - u)]


def bfs_field(gmap, sources, radius, passable_required=True):
    """BFS returning {cell: distance} and {cell: parent}."""
    dist = {}
    parent = {}
    q = deque()
    seen = set()
    for sx, sz in sources:
        if 0 <= sx < GRID and 0 <= sz < GRID and (sx, sz) not in seen:
            ok = gmap[sx, sz] == PASSABLE if passable_required else True
            if ok:
                seen.add((sx, sz))
                dist[(sx, sz)] = 0
                parent[(sx, sz)] = None
                q.append((sx, sz, 0))
    while q:
        x, z, d = q.popleft()
        if d >= radius:
            continue
        nd = d + 1
        for dx, dz in DIRS:
            nx, nz = x + dx, z + dz
            if not (0 <= nx < GRID and 0 <= nz < GRID):
                continue
            if gmap[nx, nz] != PASSABLE:
                continue
            if (nx, nz) not in dist:
                dist[(nx, nz)] = nd
                parent[(nx, nz)] = (x, z)
                q.append((nx, nz, nd))
    return dist, parent


def safe_components(safe_cells, hpos):
    """Connected components over the safe subgraph; return {cell: comp_size}
    and the id of the component containing the hero."""
    comp_id = {}
    sizes = []
    cid = 0
    for start in safe_cells:
        if start in comp_id:
            continue
        q = deque([start])
        comp_id[start] = cid
        members = [start]
        while q:
            x, z = q.popleft()
            for dx, dz in DIRS:
                n = (x + dx, z + dz)
                if n in safe_cells and n not in comp_id:
                    comp_id[n] = cid
                    members.append(n)
                    q.append(n)
        sizes.append(len(members))
        cid += 1
    comp_size = {cell: sizes[comp_id[cell]] for cell in comp_id}
    hero_cid = comp_id.get(hpos, -1)
    return comp_size, hero_cid, sizes


def flash_landing(gmap, hpos, d):
    dx, dz = DIRS[d]
    length = FLASH_LEN[d]
    last = hpos
    for i in range(1, length + 1):
        nx, nz = hpos[0] + dx * i, hpos[1] + dz * i
        if 0 <= nx < GRID and 0 <= nz < GRID and gmap[nx, nz] == PASSABLE:
            last = (nx, nz)
        else:
            break
    return last, last != hpos


def save_map(gmap, path):
    with open(path, "w") as f:
        for x in range(GRID):
            row = []
            for z in range(GRID):
                v = gmap[x, z]
                row.append("#" if v == BLOCKED else ("." if v == PASSABLE else "?"))
            f.write("".join(row) + "\n")


class Planner:
    def __init__(self):
        self.gmap = np.full((GRID, GRID), UNKNOWN, dtype=np.int32)
        self.last_mon = {}      # id -> (x,z)
        self.treasures = {}     # (x,z) -> status
        self.buffs = {}         # (x,z) -> status
        self.recent = deque(maxlen=8)

    def update_world(self, obs, hpos):
        fs = obs["frame_state"]
        view = parse_view(obs.get("map_info"))
        if view is not None and hpos is not None:
            merge_view(self.gmap, view, hpos)
        for m in fs.get("monsters", []):
            mid = m.get("monster_id", -1)
            mp = get_pos(m)
            if mp is not None:
                self.last_mon[mid] = mp
        for org in fs.get("organs", []):
            op = get_pos(org)
            if op is None:
                continue
            st = org.get("status", 1)
            if org.get("sub_type") == 1:
                self.treasures[op] = st
            elif org.get("sub_type") == 2:
                self.buffs[op] = st

    def monster_sources(self):
        return list(self.last_mon.values())

    def choose(self, obs, step):
        fs = obs["frame_state"]
        hero = fs["heroes"]
        hpos = get_pos(hero)
        legal = obs.get("legal_action") or [1] * 16

        monster_speed = 2 if step >= 500 else 1
        hero_speed = 2 if hero.get("buff_remaining_time", 0) > 0 else 1

        # Fields
        hero_dist, hero_parent = bfs_field(self.gmap, [hpos], HERO_RADIUS)
        msources = self.monster_sources()
        mon_dist, _ = bfs_field(self.gmap, msources, MONSTER_RADIUS)

        def monster_arrival(cell):
            d = mon_dist.get(cell, None)
            if d is None:
                return MONSTER_RADIUS + 5.0
            return d / float(monster_speed)

        # Margins and safe set
        margin = {}
        safe = set()
        for cell, hd in hero_dist.items():
            ha = hd / float(hero_speed)
            mg = monster_arrival(cell) - ha
            margin[cell] = mg
            if mg >= SAFE_THRESH:
                safe.add(cell)

        comp_size, hero_cid, sizes = safe_components(safe, hpos)
        largest = max(sizes) if sizes else 0

        # Current danger
        cur_margin = margin.get(hpos, monster_arrival(hpos))

        # Buff / treasure targets in known world
        buff_targets = [c for c, st in self.buffs.items() if st == 1]
        treasure_targets = [c for c, st in self.treasures.items() if st == 1]

        def first_step_to(cell):
            cur = cell
            guard = 0
            while hero_parent.get(cur, None) is not None and hero_parent[cur] != hpos:
                cur = hero_parent[cur]
                guard += 1
                if guard > HERO_RADIUS:
                    return None
            for d in range(8):
                dx, dz = DIRS[d]
                if (hpos[0] + dx, hpos[1] + dz) == cur:
                    return d, cur
            return None

        # Score candidate cells
        best_cell, best_score = hpos, -1e18
        for cell, hd in hero_dist.items():
            if hd < 1:
                continue
            mg = margin[cell]
            if mg < EMERGENCY_MARGIN:
                continue
            csize = comp_size.get(cell, 0)
            score = 0.0
            # Strongly prefer the largest safe component (deep escape space)
            score += (csize / float(max(largest, 1))) * 6.0
            score += min(mg, 12.0) * 0.6
            # Stay in the same component the hero is in
            if comp_size.get(cell, 0) == largest:
                score += 1.5
            # Target guidance
            fs_step = first_step_to(cell)
            if fs_step is not None:
                first_d = fs_step[0]
                for bt in buff_targets:
                    if abs(bt[0] - cell[0]) + abs(bt[1] - cell[1]) <= 1:
                        score += 2.0
                for tt in treasure_targets:
                    if abs(tt[0] - cell[0]) + abs(tt[1] - cell[1]) <= 1 and mg >= 4:
                        score += 1.0
                # Exploration: prefer cells bordering unknown region
                x, z = cell
                if self.gmap[x, z] == PASSABLE:
                    for dx, dz in DIRS:
                        nx, nz = x + dx, z + dz
                        if 0 <= nx < GRID and 0 <= nz < GRID and self.gmap[nx, nz] == UNKNOWN:
                            score += 0.8
                            break
            # Anti-stuck and edge penalties
            if cell in self.recent:
                score -= 1.2
            for d in (cell[0], cell[1], GRID - 1 - cell[0], GRID - 1 - cell[1]):
                if d < 12:
                    score -= (12 - d) * 0.15
            if score > best_score:
                best_score, best_cell = score, cell

        # Resolve first walk action
        walk = first_step_to(best_cell)
        walk_action = walk[0] if walk is not None else None
        if walk_action is None or not legal[walk_action]:
            for d in range(8):
                if legal[d]:
                    nx, nz = hpos[0] + DIRS[d][0], hpos[1] + DIRS[d][1]
                    if 0 <= nx < GRID and 0 <= nz < GRID and self.gmap[nx, nz] == PASSABLE:
                        walk_action = d
                        break

        # Flash on emergency
        danger = cur_margin <= EMERGENCY_MARGIN
        no_escape = comp_size.get(hpos, 0) < 3 and cur_margin < SAFE_THRESH
        if (danger or no_escape) and sum(legal[8:16]) > 0 and hero.get("flash_cooldown", 0) == 0:
            best_d, best_md = None, -1e18
            for d in range(8):
                if not legal[8 + d]:
                    continue
                land, moved = flash_landing(self.gmap, hpos, d)
                if not moved:
                    continue
                md = monster_arrival(land)
                # landing must have an escape
                esc = 0
                for dx, dz in DIRS:
                    nx, nz = land[0] + dx, land[1] + dz
                    if 0 <= nx < GRID and 0 <= nz < GRID and self.gmap[nx, nz] == PASSABLE:
                        esc += 1
                if esc < 2:
                    continue
                if md > best_md:
                    best_md, best_d = md, d
            if best_d is not None and best_md > cur_margin + 2.0:
                self.recent.append(hpos)
                return 8 + best_d, f"flash d{best_d} curM={cur_margin:.1f}"

        if walk_action is None:
            walk_action = 0
        self.recent.append(hpos)
        why = f"target={best_cell} m={margin.get(best_cell, 0):.1f} csize={comp_size.get(best_cell,0)}"
        return walk_action, why


def run_episode(env, uc, planner, ei):
    env_obs = env.reset(usr_conf=uc)
    step = 0
    t0 = time.time()
    while True:
        obs = env_obs["observation"]
        hpos = get_pos(obs["frame_state"]["heroes"])
        planner.update_world(obs, hpos)
        action, why = planner.choose(obs, step)

        if step < 20 or step % 50 == 0:
            hero = obs["frame_state"]["heroes"]
            out(f"TRACE e{ei} s={step} hero={hpos} act={action} "
                f"buff={hero.get('buff_remaining_time',0)} cd={hero.get('flash_cooldown',0)} "
                f"mon={planner.monster_sources()} {why}")

        env_reward, env_obs = env.step(action)
        if "observation" not in env_obs:
            out(f"e{ei} step error at s={step}: {env_obs.get('extra_info')}")
            return step
        step += 1
        if env_obs.get("terminated") or env_obs.get("truncated"):
            out(f"EPISODE e{ei} END s={step} elapsed={time.time()-t0:.1f}s")
            save_map(planner.gmap, f"/tmp/gmap_e{ei}.txt")
            out(f"MAP e{ei} passable={int((planner.gmap==PASSABLE).sum())} "
                f"blocked={int((planner.gmap==BLOCKED).sum())}")
            return step
        if step >= 1000:
            out(f"EPISODE e{ei} WIN reached 1000 elapsed={time.time()-t0:.1f}s")
            save_map(planner.gmap, f"/tmp/gmap_e{ei}.txt")
            return step


def workflow(envs, agents, logger=None, monitor=None, *a, **k):
    out("PROBE v25 START")
    uc = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    results = []
    for ei in range(len(envs)):
        try:
            planner = Planner()
            s = run_episode(envs[ei], uc, planner, ei)
            results.append(s)
        except Exception:
            import traceback
            out("CRASH:\n" + traceback.format_exc())
    out(f"RESULTS survival steps: {results}")
    out("PROBE v25 END")
    time.sleep(3)
