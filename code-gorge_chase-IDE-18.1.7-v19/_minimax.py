#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
One-step lookahead minimax flee policy in an open grid.
开阔地单步 lookahead minimax 逃跑策略。

For each legal hero move, simulate the monster's greedy best response, then
pick the hero move that maximizes the resulting distance (with an edge penalty).
This is near-optimal local play; if it still dies quickly, the open grid with no
buff is an essentially losing position for the hero.
"""
import math
import random
import numpy as np

SIZE = 128
MAX_STEP = 1000
DIRS = m2_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def ok(x, z):
    return 0 <= x < SIZE and 0 <= z < SIZE


def edge_pen(x, z):
    EDGE = 22
    pen = 0.0
    for d in (x, z, SIZE - 1 - x, SIZE - 1 - z):
        if d < EDGE:
            pen += (EDGE - d) / EDGE
    return pen


def mon_best(mx, mz, hx, hz):
    """Monster greedy response: neighbour minimizing Euclidean distance."""
    best, bd = (mx, mz), 1e18
    for dx, dz in DIRS:
        nx, nz = mx + dx, mz + dz
        if ok(nx, nz):
            d = math.hypot(nx - hx, nz - hz)
            if d < bd:
                bd, best = d, (nx, nz)
    return best


def run(seed=0):
    rng = random.Random(seed)
    hx = hz = SIZE // 2
    ang = rng.uniform(0, 2 * math.pi)
    mx = int(hx + 10 * math.cos(ang))
    mz = int(hz + 10 * math.sin(ang))
    mx = min(max(mx, 1), SIZE - 2)
    mz = min(max(mz, 1), SIZE - 2)
    mspd = 1
    for step in range(1, MAX_STEP + 1):
        # hero chooses move maximizing post-response distance (with edge penalty)
        best_score, bestmove = -1e18, None
        for dx, dz in DIRS:
            nx, nz = hx + dx, hz + dz
            if not ok(nx, nz):
                continue
            # monster response
            rmx, rmz = mon_best(mx, mz, nx, nz)
            d = math.hypot(rmx - nx, rmz - nz)
            score = d - edge_pen(nx, nz) * 2.0
            if score > best_score:
                best_score, bestmove = score, (nx, nz)
        if bestmove is None:
            return step
        hx, hz = bestmove
        mx, mz = mon_best(mx, mz, hx, hz)
        if math.hypot(mx - hx, mz - hz) <= 1.5:
            return step
    return MAX_STEP


if __name__ == "__main__":
    res = [run(s) for s in range(16)]
    a = np.array(res)
    print(f"minimax open no-buff: {res}")
    print(f"mean {a.mean():.0f} median {np.median(a):.0f} min {a.min()} max {a.max()}")
