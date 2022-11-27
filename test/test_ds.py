from gtclu.test.util import read_labels, gen_paras, read_data
import timeit

from sklearn.metrics import rand_score
from pyclustering.cluster.dbscan import dbscan
import sys

sys.setrecursionlimit(10000)


datasets = ["50k-20d-10c"]

E = gen_paras(0.1, 3, 0.1)
M = gen_paras(5, 5, 1)
for dataset in datasets:
    sf = "/home/doors/Code/dataset/efficiency/" + dataset
    cf = "/home/doors/Code/dataset/efficiency/" + dataset + "-class"
    labels = read_labels(cf)
    max_ari = -1
    max_metircs = ()
    max_paras = ()
    time_cost = 0
    data = read_data(sf)
    for e in E:
        for m in M:
            start = timeit.default_timer()
            db = dbscan(data, e, m, ccore=False)
            db.process()
            clus = db.get_clusters()
            end = timeit.default_timer()

            prelabels = [0] * len(data)

            for i, v in enumerate(clus):
                for j in v:
                    prelabels[j] = i + 1

            ri = rand_score(labels, prelabels)
            print(dataset, ri, end - start, e, m)

            # ami = adjusted_mutual_info_score(labels, prelabels)
            # ari = adjusted_rand_score(labels, prelabels)
            # pur = purity(labels, prelabels)

            # if ari > max_ari:
            #    max_ari = ari
            #    max_metircs = (ari, ami, pur)
            #    max_paras = (e, m)
            #    time_cost = end - start
            # print(max_metircs, max_paras, (ari), (e, m))
    # print(
    #    dataset,
    #    max_metircs[0],
    #    max_metircs[1],
    #    max_metircs[2],
    #    max_paras[0],
    #    max_paras[1],
    # )
