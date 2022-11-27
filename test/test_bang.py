from gtclu.test.util import read_labels, gen_paras, purity, read_data
import timeit

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, rand_score
from pyclustering.cluster.bang import bang
import sys

sys.setrecursionlimit(10000)


datasets = ["50k-20d-10c"]

E = gen_paras(0.01, 2, 0.001)
M = gen_paras(1, 100, 1)
for dataset in datasets:
    sf = "/home/doors/Code/dataset/efficiency/" + dataset
    cf = "/home/doors/Code/dataset/efficiency/" + dataset + "-class"

    labels = read_labels(cf)
    max_ari = -1
    max_metircs = ()
    max_paras = ()
    time_cost = 0
    data = read_data(sf)
    for m in M:
        start = timeit.default_timer()
        op = bang(data, m)
        op.process()
        clus = op.get_clusters()
        end = timeit.default_timer()

        prelabels = [-1] * len(data)

        for i, v in enumerate(clus):
            for j in v:
                prelabels[j] = i

        ri = rand_score(labels, prelabels)
        print(dataset, ri, end - start, m)

    #    ami = adjusted_mutual_info_score(labels, prelabels)
    #    ari = adjusted_rand_score(labels, prelabels)
    #    pur = purity(labels, prelabels)

    #    if ari > max_ari:
    #        max_ari = ari
    #        max_metircs = (ari, ami, pur)
    #        max_paras = m
    #        time_cost = end - start
    # print(
    #    dataset,
    #    max_metircs[0],
    #    max_metircs[1],
    #    max_metircs[2],
    #    max_paras,
    # )
