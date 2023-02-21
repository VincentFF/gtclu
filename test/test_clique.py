from gtclu.test.util import read_labels, gen_paras, read_data, purity
import timeit

from sklearn.metrics import rand_score, adjusted_mutual_info_score, adjusted_rand_score
from pyclustering.cluster.clique import clique
import sys
import math

sys.setrecursionlimit(50000)


datasets = [
    "10k-10d-10c-0.6",
    # "letter",
]

E = gen_paras(2, 4, 1)
#M = gen_paras(20, 20, 1)
M = [5]
for dataset in datasets:
    sf = "/home/doors/Code/dataset/efficiency/" + dataset
    cf = "/home/doors/Code/dataset/efficiency/" + dataset + "-class"

    #labels = read_labels(cf)
    max_ari = -1
    max_metircs = ()
    max_paras = ()
    time_cost = 0
    data = read_data(sf)
        # width = e / math.sqrt(len(data[0]))
        # e = int(math.floor(1 / width))
    for m in M:
        for e in E:
            start = timeit.default_timer()
            op = clique(data, e, m, ccore=False)
            op.process()
            clus = op.get_clusters()
            end = timeit.default_timer()
            print(e,m ,end - start)

    #        prelabels = [-1] * len(data)

    #        for i, v in enumerate(clus):
    #            for j in v:
    #                prelabels[j] = i

    #        # ri = rand_score(labels, prelabels)
    #        # print(dataset, ri, end - start, m)
    #        # print(prelabels)
    #        ami = adjusted_mutual_info_score(labels, prelabels)
    #        ari = adjusted_rand_score(labels, prelabels)
    #        pur = purity(labels, prelabels)
    #        print(ari,e,m)
    #        if ari > max_ari:
    #            max_ari = ari
    #            max_metircs = (ari, ami, pur)
    #            max_paras = (e, m)
    #            time_cost = end - start
    #        print(
    #            dataset,
    #            max_metircs[0],
    #            max_metircs[1],
    #            max_metircs[2],
    #            max_paras[0],
    #            max_paras[1],
    #        )
    # print(
    #    dataset,
    #    max_metircs[0],
    #    max_metircs[1],
    #    max_metircs[2],
    #    max_paras[0],
    #    max_paras[1],
    # )
