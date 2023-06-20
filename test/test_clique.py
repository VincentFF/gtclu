from gtclu.test.util import read_labels, gen_paras, read_data
import timeit

from sklearn.metrics import rand_score
from pyclustering.cluster.clique import clique, clique_visualizer
import sys

sys.setrecursionlimit(50000)


datasets = [
    "D31",
    # "letter",
]

E = gen_paras(48, 100, 1)
M = gen_paras(10, 10, 1)
# M = [0]
for dataset in datasets:
    sf = "/home/doors/Code/dataset/quality/" + dataset
    cf = "/home/doors/Code/dataset/quality/" + dataset + "-class"

    labels = read_labels(cf)
    data = read_data(sf)
    max_ri = -1
    max_e, max_m = 0, 0
    for e in E:
        for m in M:
            start = timeit.default_timer()
            op = clique(data, e, m, ccore=True)
            op.process()
            clus = op.get_clusters()
            end = timeit.default_timer()
            # print(e, m, end - start)

            prelabels = [-1] * len(data)

            for i, v in enumerate(clus):
                for j in v:
                    prelabels[j] = i

            clique_visualizer.show_clusters(data, clus, op.get_noise())
            ri = rand_score(labels, prelabels)
            if ri > max_ri:
                max_ri = ri
                max_e, max_m = e, m
            print(e, m, "ri: ", ri, max_ri)
    print(max_e, max_m, max_ri)
