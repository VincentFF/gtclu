from gtclu.test.util import gen_paras, read_data
import timeit
from pyclustering.cluster.optics import optics
import sys
import math

sys.setrecursionlimit(50000)


datasets = [
    "10k-10d-5c",
    # "letter",
]

E = gen_paras(0.2, 2, 0.2)
M = gen_paras(5, 5, 1)
for dataset in datasets:
    sf = "/home/doors/Code/dataset/efficiency/" + dataset
    data = read_data(sf)
    for e in E:
        for m in M:
            start = timeit.default_timer()
            db = optics(data, e, m, ccore=False)
            db.process()
            clus = db.get_clusters()
            end = timeit.default_timer()
            print(dataset, end - start, e, m)
