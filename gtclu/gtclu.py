from collections import deque
import math
import copy
import numpy as np


class GTCLU:
    def __init__(self, epsilon, minpts, d, algo="bfs", tree_level=10):
        self.epsilon = epsilon
        self.minpts = minpts
        self.d = d

        self.algo = algo

        self.table = {}
        self.cores = []
        self.borders = []
        self.edges = {}
        self.flag = 0

        self.width = epsilon / math.sqrt(d)
        self.coord_split = int(math.floor(1 / self.width))
        if algo == "bfs":
            self.directions = self._directions(d)
        else:
            self.tree_level = tree_level

        self.clusters = []

    def learn_one(self, x):
        """receive a point, put it into the table
        if a table grid's weight more than minpts, add the grid to core list.
        Args:
            x (np.ndarray): a d-dimension point
        """
        pos = self._get_pos(x)
        if pos not in self.table:
            self.table[pos] = Grid(pos)
        self.table[pos].add_point(x)

    def fit(self):
        if self.algo == "bfs":
            self._bfs_clustering()
        elif self.algo == "tree":
            # print("building tree")
            tree = GridTree(
                self.table,
                self.d,
                self.minpts,
                self.epsilon,
                0,
                self.coord_split,
                level=self.tree_level,
            )
            tree.fit()
            self.clusters = tree.clusters

        else:
            raise ValueError("Unknown algorithm parameter: {}".format(self.algo))

    def _check_cores(self):
        for pos, grid in self.table.items():
            if grid.weight >= self.minpts:
                self.cores.append(pos)
                grid.core = True
            else:
                near_weight = 0
                self.edges[pos] = []

                center = grid.center
                for d in self.directions:
                    tem_pos = tuple([pos[i] + d[i] for i in range(self.d)])
                    if tem_pos in self.table:
                        tem_grid = self.table[tem_pos]
                        near_weight += tem_grid.weight * min(
                            0.5
                            * self.epsilon
                            / np.linalg.norm(center - tem_grid.center),
                            1,
                        )
                        self.edges[pos].append(tem_pos)
                if near_weight + grid.weight >= self.minpts:
                    self.cores.append(pos)
                    grid.core = True
                else:
                    self.borders.append(pos)

    def _bfs_clustering(self):
        """BFS merging grids strategy for clustering"""

        self._check_cores()

        # clustering cores
        for pos in self.cores:
            grid = self.table[pos]
            if grid.cluster != -1:
                continue
            que = deque()
            grid.core = self.flag
            que.append(pos)

            while que:
                cur_pos = que.popleft()

                if cur_pos in self.edges:
                    for near_pos in self.edges[cur_pos]:
                        near_grid = self.table[near_pos]
                        if near_grid.core and near_grid.cluster == -1:
                            near_grid.cluster = self.flag
                            que.append(near_pos)
                else:
                    for direct in self.directions:
                        near_pos = tuple(
                            [cur_pos[i] + direct[i] for i in range(self.d)]
                        )
                        if near_pos not in self.table:
                            continue
                        near_grid = self.table[near_pos]
                        if near_grid.core and near_grid.cluster == -1:
                            near_grid.cluster = self.flag
                            que.append(near_pos)

            self.flag += 1

        # clustering borders
        for pos in self.borders:
            max_core_weight = 0
            max_core_pos = None
            border_grid = self.table[pos]
            if pos in self.edges:
                for near_pos in self.edges[pos]:
                    near_grid = self.table[near_pos]
                    if near_grid.core and near_grid.weight > max_core_weight:
                        max_core_pos = near_pos
                        max_core_weight = near_grid.weight
            else:
                for direct in self.directions:
                    near_pos = tuple([pos[i] + direct[i] for i in range(self.d)])
                    if near_pos not in self.table:
                        continue
                    near_grid = self.table[near_pos]
                    if near_grid.core and near_grid.weight > max_core_weight:
                        max_core_pos = near_pos
                        max_core_weight = near_grid.weight

            if max_core_pos:
                border_grid.cluster = self.table[max_core_pos].cluster

    def predict_one(self, x):
        pos = self._get_pos(x)
        return self.table[pos].cluster if pos in self.table else -1

    def _get_pos(self, x):
        pos = np.clip(np.floor(self.coord_split * x), 0, self.coord_split - 1)
        return tuple(pos.astype(int))

    def _directions(self, d):
        """Generate near grids directions"""
        pos = [0, -1, 1]
        res = [[0 for i in range(d)]]
        for i in range(d):
            new = []
            for dire in res:
                for j in pos:
                    tem = copy.copy(dire)
                    tem[i] = j
                    new.append(tem)
            res = new
        return res[1:]

    def _tree_clustering(self):
        pass


class Grid:
    def __init__(self, pos):
        self.pos = pos
        self.weight = 0
        self.linear_sum = np.zeros(len(pos))
        self.core = False
        self.cluster = -1
        self.extra_weight = 0

    def add_point(self, x):
        self.weight += 1
        self.linear_sum += x

    @property
    def center(self):
        if self.weight == 0:
            return np.array(self.pos)
        return self.linear_sum / self.weight

    def __str__(self):
        return "pos: {0}, weight: {1}, center: {2}".format(
            self.pos, self.weight, self.center
        )

    def __repr__(self):
        return self.__str__()


class GridTree:
    def __init__(
        self, table, dimension, min_pts, epsilon, dim_lower, dim_high, level=10
    ):
        """A grid tree for parallel gbscan algorithm
        Args:
            table: a dict of grid table
            dim_lower,dim_high: 0 and coord_split number
        """
        self.table = table
        self.epsilon = epsilon
        self.threshold = epsilon  # threshold for merging grids
        self.dimension = dimension
        # self.grid_width = grid_width
        # It depends on the design of the grid splitting, here we use 1
        self.grid_width = 1
        self.min_pts = min_pts
        self.level = level
        self.level_dim = self._cal_level_dim()

        self.level_nodes = [[] for _ in range(self.level + 1)]
        self.grids = list(table.values())
        self.dim_bounds = [[dim_lower, dim_high] for _ in range(self.dimension)]
        self.root = self._build_tree(0, self.grids, self.dim_bounds)
        self.clusters = []

    def _cal_level_dim(self):
        """Calculate the level of the grid tree
        Let level = k+1,
        where k is the minimum integer that 2^k >= cpu_nums
        """
        level = self.level

        level_dim = []
        if level > self.dimension:
            for i in range(level + 1):
                level_dim.append(i % self.dimension)
        else:
            for i in range(level + 1):
                level_dim.append(i)
        return level_dim

    def _build_tree(self, which_level, grids, dim_bounds):
        """Build the grid tree
        Args:
            which_level: the level of the tree
        """
        # print("building tree level: ", which_level)
        if not grids:
            return None

        # reach to the leaf level, return leaf node
        if which_level == self.level:
            leaf_node = TreeNode(self.level_dim[which_level], grids)
            self.level_nodes[which_level].append(leaf_node)
            return leaf_node

        l_edges, r_edges = {}, {}
        l_grids, r_grids = [], []
        which_dim = self.level_dim[which_level]
        edge = (dim_bounds[which_dim][0] + dim_bounds[which_dim][1]) // 2
        for grid in grids:
            if grid.pos[which_dim] <= edge:
                l_grids.append(grid)
                if grid.pos[which_dim] == edge:
                    l_edges[grid.pos] = []
            else:
                r_grids.append(grid)
                if grid.pos[which_dim] == edge + 1:
                    r_edges[grid.pos] = []

        # get extra-weights and edge-neighbors of edge-grids
        for pos1 in l_edges:
            grid_1 = self.table[pos1]
            for pos2 in r_edges:
                grid_2 = self.table[pos2]
                dis = np.linalg.norm(grid_1.center - grid_2.center)
                if dis <= self.threshold:
                    sigma = min(0.5 * self.epsilon / dis, 1)
                    grid_1.extra_weight += sigma * grid_2.weight
                    grid_2.extra_weight += sigma * grid_1.weight
                    l_edges[grid_1.pos].append(grid_2)
                    r_edges[grid_2.pos].append(grid_1)

        # non-leaf node
        node = TreeNode(which_dim)
        node.l_edges, node.r_edges = l_edges, r_edges
        l_dim_bounds = copy.deepcopy(dim_bounds)
        r_dim_bounds = copy.deepcopy(dim_bounds)
        l_dim_bounds[which_dim][1] = edge
        r_dim_bounds[which_dim][0] = edge + 1

        l_node = self._build_tree(which_level + 1, l_grids, l_dim_bounds)
        r_node = self._build_tree(which_level + 1, r_grids, r_dim_bounds)

        node.l_node = l_node
        node.r_node = r_node
        self.level_nodes[which_level].append(node)
        return node

    def fit(self):
        """Cluster the grid tree"""

        # 1. cluster the leaf nodes
        # print("cluster leaves")
        self.cluster_leaves()

        # 2. cluster the non-leaf nodes
        # print("cluster non-leaves")
        for level in range(self.level - 1, -1, -1):
            for node in self.level_nodes[level]:
                self._merge_children(node)

        self.clusters = self.root.clusters

    def cluster_leaves(self):
        """Cluster the leaf nodes"""
        for node in self.level_nodes[self.level]:
            self._cluster_leaf_node(node)

    def _cluster_leaf_node(self, node):
        """BFS cluster a leaf node"""
        clusters = []  # [{grid1,gird2...}, {grid3,grid4...},...]
        grids = node.grids
        flag = 0

        cores = []
        borders = []
        near_grids = {g.pos: [] for g in grids}
        # 1. get core grids and border grids
        for i in range(len(grids)):
            grid_1 = grids[i]
            for j in range(i + 1, len(grids)):
                grid_2 = grids[j]
                # cache the near grids
                dis = np.linalg.norm(grid_1.center - grid_2.center)
                if dis <= self.threshold:
                    sigma = min(0.5 * self.epsilon / dis, 1)
                    grid_1.extra_weight += sigma * grid_2.weight
                    grid_2.extra_weight += sigma * grid_1.weight
                    near_grids[grid_1.pos].append(grid_2)
                    near_grids[grid_2.pos].append(grid_1)
            if grid_1.weight + grid_1.extra_weight >= self.min_pts:
                cores.append(grid_1)
                grid_1.core = True
            else:
                borders.append(grid_1)

        # 2. cluster the core grids
        for grid in cores:
            if grid.cluster != -1:
                continue
            que = deque([grid])
            clusters.append(set())
            while que:
                lgrid = que.popleft()
                clusters[flag].add(lgrid)
                lgrid.cluster = flag
                for near_gird in near_grids[lgrid.pos]:
                    if near_gird.core and near_gird.cluster == -1:
                        que.append(near_gird)
            flag += 1

        # 3. cluster the border grids
        for grid in borders:
            max_weigh = 0
            max_grid = None
            for near_grid in near_grids[grid.pos]:
                if near_grid.core and near_grid.weight > max_weigh:
                    max_weigh = near_grid.weight
                    max_grid = near_grid
            if max_grid is not None:
                grid.cluster = max_grid.cluster
                clusters[max_grid.cluster].add(grid)

        # 4. assign the clusters to the node
        node.clusters = clusters

    def _merge_children(self, node):
        """Merge the children nodes of node"""
        if not node:
            return

        if not node.l_node or not node.r_node:
            if not node.l_node:
                node.clusters = node.r_node.clusters
            else:
                node.clusters = node.l_node.clusters
            return

        l_clusters = node.l_node.clusters
        r_clusters = node.r_node.clusters
        l_edges = node.l_edges
        r_edges = node.r_edges

        # 1. merge noise grids in l_edges to r_clusters,
        # and remove them from l_edges
        l_noise_poses = []
        for pos in l_edges:
            if self.table[pos].cluster != -1:  # not noise
                continue
            l_noise_poses.append(pos)
            for near_grid in l_edges[pos]:
                if near_grid.core:
                    r_clusters[near_grid.cluster].add(self.table[pos])
                    break
        for noise_pos in l_noise_poses:
            l_edges.pop(noise_pos)

        # 2. check which r_clusters and r_noises can be merged into l_clusters
        merge_table = {}
        for pos in r_edges:
            r_grid = self.table[pos]
            # for noise grids, merge them to l_clusters directly
            if r_grid.cluster == -1:
                for near_grid in r_edges[pos]:
                    if near_grid.core:
                        l_clusters[near_grid.cluster].add(r_grid)
                        break
            # for core grids, make a merge table for merging later
            else:
                if not r_grid.core:  # border grid cant be merged
                    continue
                for near_grid in r_edges[pos]:
                    if near_grid.core:
                        if r_grid.cluster in merge_table:
                            merge_table[r_grid.cluster].add(near_grid.cluster)
                        else:
                            merge_table[r_grid.cluster] = {near_grid.cluster}
        for c in merge_table:
            merge_table[c] = list(merge_table[c])

        # 3. handle the merge table to get the final merged clusters

        # 3.1 merge right clusters to left clusters first
        remove_right_clusters = set()
        for r_clu, l_clus in merge_table.items():
            l_clusters[l_clus[0]].update(r_clusters[r_clu])
            remove_right_clusters.add(r_clu)

        # 3.2 then merge left clusters if possible
        # some left clusters can be merged anew because the
        # right clusters joined

        # firstly, use a disjoint set to mark that how to merge the clusters
        disjoint_set = [i for i in range(len(l_clusters))]
        for clus in merge_table.values():

            first_root = clus[0]
            while first_root != disjoint_set[first_root]:
                first_root = disjoint_set[first_root]

            for i in range(1, len(clus)):
                now_root = clus[i]
                while now_root != disjoint_set[now_root]:
                    now_root = disjoint_set[now_root]
                disjoint_set[now_root] = first_root

        # secondly, merge the clusters according to the disjoint set
        for i in range(len(disjoint_set)):
            now_root = i
            while now_root != disjoint_set[now_root]:
                now_root = disjoint_set[now_root]
            if now_root != i:  # should be merged
                l_clusters[now_root].update(l_clusters[i])
                l_clusters[i] = set()

        # finally, generate the final clusters
        final_clusters = []
        for l_clu in l_clusters:
            if l_clu:
                final_clusters.append(l_clu)
        for i, r_clu in enumerate(r_clusters):
            if i not in remove_right_clusters:
                final_clusters.append(r_clu)

        # 4. relabel the grids and assign the final clusters to the node
        for i in range(len(final_clusters)):
            for grid in final_clusters[i]:
                grid.cluster = i
        node.clusters = final_clusters


class TreeNode:
    def __init__(self, which_dim, grids=None):
        self.which_dim = which_dim
        self.grids = grids  # leaf node holds the grids
        self.l_edges = {}
        self.r_edges = {}  # non-leaf node holds the edges
        self.l_node = None
        self.r_node = None
        self.clusters = []


class Cluster:
    def __init__(self, pos_set, label):
        self.pos_set = pos_set
        self.label = label
