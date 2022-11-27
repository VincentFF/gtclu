from gtclu.test.util import read_labels, gen_paras, purity, read_data
import timeit

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from pyclustering.cluster.optics import optics
import sys

sys.setrecursionlimit(10000)


datasets = [
    "Aggregation",
    "Compound",
    "D31",
    "flame",
    "g2-2-30",
    "g2-4-30",
    "g2-8-30",
    "iris",
    "jain",
    "letter",
    "pathbased",
    "R15",
    "s1",
    "s2",
    "s3",
    "s4",
    "skewed",
    "spiral",
    "unbalance2",
]

for dataset in datasets:

    sf = "/home/doors/Code/dataset/quality/" + dataset
    cf = "/home/doors/Code/dataset/quality/" + dataset + "-class"

    E = gen_paras(0.01, 0.2, 0.001)
    M = gen_paras(1, 30, 1)

    labels = read_labels(cf)
    max_ari = -1
    max_metircs = ()
    max_paras = ()
    time_cost = 0
    data = read_data(sf)
    for e in E:
        for m in M:
            start = timeit.default_timer()
            op = optics(data, e, m, ccore=False)
            op.process()
            clus = op.get_clusters()
            end = timeit.default_timer()

            prelabels = [0] * len(data)

            for i, v in enumerate(clus):
                for j in v:
                    prelabels[j] = i + 1

            ami = adjusted_mutual_info_score(labels, prelabels)
            ari = adjusted_rand_score(labels, prelabels)
            pur = purity(labels, prelabels)

            if ari > max_ari:
                max_ari = ari
                max_metircs = (ari, ami, pur)
                max_paras = (e, m)
                time_cost = end - start
            # print(max_metircs, max_paras, (ari), (e, m))
    print(
        dataset,
        max_metircs[0],
        max_metircs[1],
        max_metircs[2],
        max_paras[0],
        max_paras[1],
    )
