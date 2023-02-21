from gtclu.test.util import gen_paras, read_data
import timeit
from gtclu.gtclu.gtclu_fast import GTCLU
import sys
import math

sys.setrecursionlimit(50000)


datasets = [
    # "10k-10d-5c",
    "10k-10d-10c-0.6",
]

E = gen_paras(2, 10, 1)
M = [5,20,50]
for dataset in datasets:
    sf = "/home/doors/Code/dataset/efficiency/" + dataset
    data = read_data(sf)
    d = len(data[0])
    for m in M:
        for e in E:
            ep = math.sqrt(d)/e
            start = timeit.default_timer()
            gtclu = GTCLU(ep, m, d, algo="tree")
            for p in data:
                gtclu.learn_one(p)
            gtclu.fit()
            end = timeit.default_timer()
            print(e, m,end - start)
