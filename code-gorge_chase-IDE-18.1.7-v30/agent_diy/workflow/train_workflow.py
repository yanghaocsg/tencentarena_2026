#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""v26: robust safety-distance planner with move-failure feedback.

Fixes v25's freeze (gmap said a cell was passable but env.step did not
move the hero, and the planner kept requesting the same action):

1. After every walk action we feed back whether the hero actually moved.
   A (position, action) that fails to move is recorded in failed_moves and
   never chosen again at that position.
2. Stuck detection rotates the tie-breaking start direction; first stuck
   event dumps the raw 3x3 view alongside the gmap 3x3 to expose any
   view/gmap disagreement.
3. Actions are chosen among the 8 direct neighbours (each scored by the
   size of its safe connected component + margin + target guidance),
   which is more stable than steering toward a far target.
4. Runs several episodes on envs[0].
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
N_EPISODES = 2
STUCK_LIMIT = 2


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


def bfs_field(gmap, sources, radius):
    dist, parent = {}, {}
    q, seen = deque(), set()
    for sx, sz in sources:
        if 0 <= sx < GRID and 0 <= sz < GRID and (sx, sz) not in seen \
                and gmap[sx, sz] == PASSABLE:
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
            if 0 <= nx < GRID and 0 <= nz < GRID \
                    and gmap[nx, nz] == PASSABLE and (nx, nz) not in dist:
                dist[(nx, nz)] = nd
                parent[(nx, nz)] = (x, z)
                q.append((nx, nz, nd))
    return dist, parent


def safe_components(safe_cells, hpos):
    comp_id, sizes, cid = {}, [], 0
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
    return comp_size, (max(sizes) if sizes else 0)


def flash_landing(gmap, hpos, d):
    dx, dz = DIRS[d]
    last = hpos
    for i in range(1, FLASH_LEN[d] + 1):
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
    def __init__(self, ei):
        self.ei = ei
        self.gmap = np.full((GRID, GRID), UNKNOWN, dtype=np.int32)
        self.last_mon = {}
        self.treasures = {}
        self.buffs = {}
        self.recent = deque(maxlen=8)
        self.failed_moves = set()
        self.stuck_count = 0
        self.start_dir = 0
        self.diag_done = False

    def update_world(self, obs, hpos):
        fs = obs["frame_state"]
        view = parse_view(obs.get("map_info"))
        if view is not None and hpos is not None:
            merge_view(self.gmap, view, hpos)
        for m in fs.get("monsters", []):
            mp = get_pos(m)
            if mp is not None:
                self.last_mon[m.get("monster_id", -1)] = mp
        for org in fs.get("organs", []):
            op = get_pos(org)
            if op is None:
                continue
            if org.get("sub_type") == 1:
                self.treasures[op] = org.get("status", 1)
            elif org.get("sub_type") == 2:
                self.buffs[op] = org.get("status", 1)
        return view

    def feedback(self, old_pos, action, new_pos):
        if action < 8 and new_pos == old_pos:
            self.failed_moves.add((old_pos[0], old_pos[1], action))
            self.stuck_count += 1
        else:
            self.stuck_count = 0

    def choose(self, obs, step, raw_view):
        fs = obs["frame_state"]
        hero = fs["heroes"]
        hpos = get_pos(hero)
        legal = obs.get("legal_action") or [1] * 16

        monster_speed = 2 if step >= 500 else 1
        hero_speed = 2 if hero.get("buff_remaining_time", 0) > 0 else 1

        hero_dist, _ = bfs_field(self.gmap, [hpos], HERO_RADIUS)
        mon_dist, _ = bfs_field(self.gmap, list(self.last_mon.values()), MONSTER_RADIUS)

        def mon_arrival(cell):
            d = mon_dist.get(cell)
            return (MONSTER_RADIUS + 5.0) if d is None else d / float(monster_speed)

        margin, safe = {}, set()
        for cell, hd in hero_dist.items():
            mg = mon_arrival(cell) - hd / float(hero_speed)
            margin[cell] = mg
            if mg >= SAFE_THRESH:
                safe.add(cell)
        comp_size, largest = safe_components(safe, hpos)
        cur_margin = margin.get(hpos, mon_arrival(hpos))

        buff_targets = [c for c, s in self.buffs.items() if s == 1]
        treasure_targets = [c for c, s in self.treasures.items() if s == 1]

        # Stuck diagnostics
        if self.stuck_count >= STUCK_LIMIT and not self.diag_done:
            out(f"STUCK e{self.ei} s={step} hero={hpos} start_dir={self.start_dir}")
            if raw_view is not None:
                for r in range(9, 12):
                    out(f"  rawview r{r} = {[int(v) for v in raw_view[r, 9:12]]}")
            for r in range(hpos[0] - 1, hpos[0] + 2):
                out(f"  gmap x={r} z{hpos[1]-1}-{hpos[1]+1} = "
                    f"{[int(self.gmap[r, z]) for z in range(hpos[1]-1, hpos[1]+2)]}")
            out(f"  monsters={list(self.last_mon.values())} failed="
                f"{[a for (x,z,a) in self.failed_moves if (x,z)==hpos]}")
            self.diag_done = True
        if self.stuck_count >= STUCK_LIMIT:
            self.start_dir = (self.start_dir + 1) % 8
            self.stuck_count = 0

        # Score the 8 direct neighbours
        best_d, best_score = None, -1e18
        order = [(self.start_dir + i) % 8 for i in range(8)]
        for d in order:
            if not legal[d]:
                continue
            dx, dz = DIRS[d]
            n = (hpos[0] + dx, hpos[1] + dz)
            if not (0 <= n[0] < GRID and 0 <= n[1] < GRID):
                continue
            if self.gmap[n] != PASSABLE:
                continue
            if (hpos[0], hpos[1], d) in self.failed_moves:
                continue
            mg = margin.get(n, mon_arrival(n) - 1.0 / hero_speed)
            csize = comp_size.get(n, 0)
            score = (csize / float(max(largest, 1))) * 6.0 + min(mg, 12.0) * 0.6
            if csize == largest:
                score += 1.5
            for bt in buff_targets:
                if n == bt:
                    score += 2.5
            for tt in treasure_targets:
                if n == tt and mg >= 4:
                    score += 1.2
            # exploration: neighbour bordering unknown
            for dx2, dz2 in DIRS:
                ax, az = n[0] + dx2, n[1] + dz2
                if 0 <= ax < GRID and 0 <= az < GRID and self.gmap[ax, az] == UNKNOWN:
                    score += 0.7
                    break
            if n in self.recent:
                score -= 1.0
            for dd in (n[0], n[1], GRID - 1 - n[0], GRID - 1 - n[1]):
                if dd < 12:
                    score -= (12 - dd) * 0.15
            if score > best_score:
                best_score, best_d = score, d

        # Flash emergency
        danger = cur_margin <= EMERGENCY_MARGIN
        no_escape = comp_size.get(hpos, 0) < 3 and cur_margin < SAFE_THRESH
        if (danger or no_escape) and sum(legal[8:16]) > 0 and hero.get("flash_cooldown", 0) == 0:
            best_f, best_md = None, -1e18
            for d in range(8):
                if not legal[8 + d]:
                    continue
                land, moved = flash_landing(self.gmap, hpos, d)
                if not moved:
                    continue
                esc = sum(1 for dx, dz in DIRS
                          if self.gmap[land[0] + dx, land[1] + dz] == PASSABLE)
                if esc < 2:
                    continue
                md = mon_arrival(land)
                if md > best_md:
                    best_md, best_f = md, d
            if best_f is not None and best_md > cur_margin + 2.0:
                self.recent.append(hpos)
                return 8 + best_f, f"flash d{best_f} curM={cur_margin:.1f}"

        if best_d is None:
            # fallback: any legal move into passable/unknown not failed
            for d in order:
                if legal[d] and (hpos[0], hpos[1], d) not in self.failed_moves:
                    best_d = d
                    break
            if best_d is None:
                best_d = 0
        self.recent.append(hpos)
        n = (hpos[0] + DIRS[best_d][0], hpos[1] + DIRS[best_d][1])
        return best_d, f"n={n} m={margin.get(n, 0):.1f} csize={comp_size.get(n,0)}"


def run_episode(env, uc, ei):
    planner = Planner(ei)
    env_obs = env.reset(usr_conf=uc)
    step = 0
    t0 = time.time()
    while True:
        obs = env_obs["observation"]
        hpos = get_pos(obs["frame_state"]["heroes"])
        raw_view = planner.update_world(obs, hpos)
        action, why = planner.choose(obs, step, raw_view)

        # Dump every monster's full field each step (focus: out-of-view signals)
        monsters = obs["frame_state"].get("monsters", [])
        for m in monsters:
            out(f"  MON id={m.get('monster_id')} view={m.get('is_in_view')} "
                f"pos={m.get('pos')} l2={m.get('hero_l2_distance')} "
                f"rel={m.get('hero_relative_direction')} speed={m.get('speed')}")

        # Perpendicular test: once the monster is out of view, walk
        # perpendicular to its bearing to force the true bearing to change.
        # If rel follows, it is a live (real-time) signal.
        hero = obs["frame_state"]["heroes"]
        if not getattr(planner, "perp_started", False):
            for m in monsters:
                if m.get("is_in_view") == 0 and m.get("hero_relative_direction"):
                    mon_dir = m["hero_relative_direction"] - 1
                    planner.perp = (mon_dir + 2) % 8
                    planner.perp_steps = 0
                    planner.perp_started = True
                    out(f"PERP TEST START s={step} hero={hpos} "
                        f"rel={m['hero_relative_direction']} mon_dir={mon_dir} "
                        f"perp_dir={planner.perp}")
                    break
        if getattr(planner, "perp_started", False) and planner.perp_steps < 8:
            action = planner.perp
            planner.perp_steps += 1
            out(f"PERP MOVE s={step} hero={hpos} act={action} "
                f"rel={[m.get('hero_relative_direction') for m in monsters]}")

        if step < 15 or step % 50 == 0:
            out(f"TRACE e{ei} s={step} hero={hpos} act={action} "
                f"buff={hero.get('buff_remaining_time',0)} cd={hero.get('flash_cooldown',0)} "
                f"mon={list(planner.last_mon.values())} {why}")

        env_reward, env_obs = env.step(action)
        if "observation" not in env_obs:
            out(f"e{ei} step error at s={step}")
            return step
        new_hpos = get_pos(env_obs["observation"]["frame_state"]["heroes"])
        if getattr(planner, "perp_started", False) and planner.perp_steps <= 8:
            after_mons = env_obs["observation"]["frame_state"].get("monsters", [])
            moved = (new_hpos != hpos)
            for m in after_mons:
                out(f"PERP AFTER s={step} hero={new_hpos} moved={moved} "
                    f"view={m.get('is_in_view')} rel={m.get('hero_relative_direction')}")
            if not moved:
                planner.perp_steps = 8  # perpendicular blocked, stop test
        planner.feedback(hpos, action, new_hpos)
        step += 1
        if env_obs.get("terminated") or env_obs.get("truncated"):
            out(f"EPISODE e{ei} END s={step} elapsed={time.time()-t0:.1f}s")
            save_map(planner.gmap, f"/tmp/gmap_e{ei}.txt")
            out(f"MAP e{ei} passable={int((planner.gmap==PASSABLE).sum())} "
                f"blocked={int((planner.gmap==BLOCKED).sum())}")
            return step
        if step >= 1000:
            out(f"EPISODE e{ei} WIN 1000 elapsed={time.time()-t0:.1f}s")
            save_map(planner.gmap, f"/tmp/gmap_e{ei}.txt")
            return step


def workflow(envs, agents, logger=None, monitor=None, *a, **k):
    out("PROBE v26 START")
    uc = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    env = envs[0]
    results = []
    for ei in range(N_EPISODES):
        try:
            results.append(run_episode(env, uc, ei))
        except Exception:
            import traceback
            out("CRASH:\n" + traceback.format_exc())
        time.sleep(1)
    out(f"RESULTS survival steps: {results}")
    out("PROBE v26 END")
    time.sleep(3)
