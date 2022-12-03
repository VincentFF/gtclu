from gtclu.test.util import gen_paras, read_data
import timeit
from gtclu.gtclu.gtclu_kdtree_py import GTCLU
import sys
import math

sys.setrecursionlimit(50000)


datasets = [
    # "10k-10d-5c",
    "Aggregation",
]

E = gen_paras(0.02, 0.1, 0.001)
M = gen_paras(5, 10, 1)
for dataset in datasets:
    sf = "/home/doors/Code/dataset/quality/" + dataset
    data = read_data(sf)
    d = len(data[0])
    for e in E:
        for m in M:
            start = timeit.default_timer()
            gtclu = GTCLU(e, m, d, algo="tree")
            for p in data:
                gtclu.learn_one(p)
            gtclu.fit()
            end = timeit.default_timer()
            print(e, end - start)
