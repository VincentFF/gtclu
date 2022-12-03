from gtclu.test.util import gen_paras, read_data
import timeit
from pyclustering.cluster.clique import clique
import sys
import math

sys.setrecursionlimit(50000)


datasets = [
    "10k-10d-5c",
    # "letter",
]

E = gen_paras(2, 0.2, -0.2)
M = gen_paras(5, 5, 1)
for dataset in datasets:
    sf = "/home/doors/Code/dataset/efficiency/" + dataset
    data = read_data(sf)
    for e in E:
        width = e / math.sqrt(len(data[0]))
        sp = int(math.floor(1 / width))
        for m in M:
            start = timeit.default_timer()
            op = clique(data, sp, m, ccore=False)
            op.process()
            clus = op.get_clusters()
            end = timeit.default_timer()
            print(dataset, end - start, e, m)
