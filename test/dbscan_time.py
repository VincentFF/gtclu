from gtclu.test.util import gen_paras, read_data
import timeit
from pyclustering.cluster.dbscan import dbscan
import sys
import math

sys.setrecursionlimit(50000)


datasets = [
    "10k-3d-10c-0.6",
    # "letter",
]

E = gen_paras(0.02, 0.2, 0.02)
M = gen_paras(50, 50, 1)
for dataset in datasets:
    sf = "/home/doors/Code/dataset/efficiency/" + dataset
    data = read_data(sf)
    for e in E:
        for m in M:
            start = timeit.default_timer()
            db = dbscan(data, e, m, ccore=False)
            db.process()
            clus = db.get_clusters()
            end = timeit.default_timer()
            print(dataset, end - start, e, m)
