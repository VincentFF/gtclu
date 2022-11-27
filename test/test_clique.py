from gtclu.test.util import read_labels, gen_paras, read_data, purity
import timeit

from sklearn.metrics import rand_score, adjusted_mutual_info_score, adjusted_rand_score
from pyclustering.cluster.clique import clique
import sys
import math

sys.setrecursionlimit(50000)


datasets = [
    "g2-8-30",
    #"letter",
]

E = gen_paras(8, 8, 1)
M = gen_paras(1, 5, 1)
for dataset in datasets:
    sf = "/home/doors/Code/dataset/quality/" + dataset
    cf = "/home/doors/Code/dataset/quality/" + dataset + "-class"

    labels = read_labels(cf)
    max_ari = -1
    max_metircs = ()
    max_paras = ()
    time_cost = 0
    data = read_data(sf)
    for e in E:
        # width = e / math.sqrt(len(data[0]))
        # e = int(math.floor(1 / width))
        for m in M:
            start = timeit.default_timer()
            op = clique(data, e, m, ccore=True)
            op.process()
            clus = op.get_clusters()
            end = timeit.default_timer()

            prelabels = [-1] * len(data)

            for i, v in enumerate(clus):
                for j in v:
                    prelabels[j] = i

            # ri = rand_score(labels, prelabels)
            # print(dataset, ri, end - start, m)
            # print(prelabels)
            ami = adjusted_mutual_info_score(labels, prelabels)
            ari = adjusted_rand_score(labels, prelabels)
            pur = purity(labels, prelabels)
            if ari > max_ari:
                max_ari = ari
                max_metircs = (ari, ami, pur)
                max_paras = (e, m)
                time_cost = end - start
            print(
                dataset,
                max_metircs[0],
                max_metircs[1],
                max_metircs[2],
                max_paras[0],
                max_paras[1],
            )
    print(
        dataset,
        max_metircs[0],
        max_metircs[1],
        max_metircs[2],
        max_paras[0],
        max_paras[1],
    )
