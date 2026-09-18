import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID_SIZE = 10 # 100 cells
MAX_PATH_LEN = 20
VOCAB_SIZE = 16
MOVES = [(-1, 0, 6), (1, 0, 7), (0, -1, 8), (0, 1, 9)]

def generate_maze(grid_size: int = 10, wall_prob: float = 0.25):
    grid = np.ones((grid_size, grid_size), dtype=int)
    for r in range(grid_size):
        for c in range(grid_size):
            if random.random() < wall_prob:
                grid[r, c] = 2 # wall
    empty_cells = [(r, c) for r in range(grid_size) for c in range(grid_size) if grid[r, c] == 1]
    if len(empty_cells) < 4:
        return generate_maze(grid_size, wall_prob)
    start, goal = random.sample(empty_cells, 2)
    grid[start[0], start[1]] = 3 # start
    grid[goal[0], goal[1]] = 4 # goal

    # BFS shortest path
    q = deque([(start[0], start[1], [])])
    visited = {start}
    shortest_moves = None
    junction_map = {}

    while q:
        r, c, path = q.popleft()
        if (r, c) == goal:
            shortest_moves = path
            break
        # Count open neighbors
        open_neighbors = []
        for dr, dc, m_id in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size and grid[nr, nc] != 2:
                open_neighbors.append((nr, nc, m_id))
        is_junction = len(open_neighbors) > 2
        junction_map[(r, c)] = is_junction

        for nr, nc, m_id in open_neighbors:
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc, path + [m_id]))

    flat_grid = grid.flatten()
    if shortest_moves is None:
        target_seq = [5, 11] # <SOS>, <UNREACHABLE>
        is_solvable = False
    else:
        target_seq = [5] + shortest_moves + [10]
        is_solvable = True

    target_seq = target_seq[:MAX_PATH_LEN]
    pad_len = MAX_PATH_LEN - len(target_seq)
    target_tokens = target_seq + [0] * pad_len
    return flat_grid, target_tokens, is_solvable

print("Testing maze generation...")
g, tgt, sol = generate_maze(10)
print("Grid shape:", g.shape, "Target len:", len(tgt), "Solvable:", sol)
print("Target:", tgt[:10])
