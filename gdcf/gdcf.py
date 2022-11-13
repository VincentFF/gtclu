import math
import numpy as np


class GDCF:
    def __init__(self, e, m, d):
        self.e = e
        self.m = m
        self.d = d

        self.table = {}

        self.width = 2 * e / math.sqrt(d)
        self.coord_split = int(math.floor(1 / self.width))

    def learn_one(self, x):
        """receive a point, put it into the table
        if a table grid's weight more than minpts, add the grid to core list.
        Args:
            x (np.ndarray): a d-dimension point
        """
        pos = self._get_pos(x)
        if pos not in self.table:
            self.table[pos] = Grid(pos)
        self.table[pos].add_one()

    def _get_pos(self, x):
        pos = np.clip(np.floor(self.coord_split * x), 0, self.coord_split - 1)
        return tuple(pos.astype(int))

    def neighbour_query(self, g, B):
        pass

    def build_hgb(self):
        pass

    def fit(self):
        pass


class Grid:
    def __init__(self, pos) -> None:
        self.pos = pos
        self.weight = 0
        self.core = False
        self.cluster = -1

    def add_one(self):
        self.weight += 1
