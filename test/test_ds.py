from gtclu.test.util import read_labels, gen_paras, purity, read_data
import timeit

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from pyclustering.cluster.dbscan import dbscan
import sys

sys.setrecursionlimit(10000)

# datasets = ["skin"]
record = "/home/doors/Code/stream/sgdbscan/test/db-record"

dataset = "s4"
# e = 0.5
# m = 20


sf = "/home/doors/Code/dataset/small/" + dataset
cf = "/home/doors/Code/dataset/small/" + dataset + "-class"

E = gen_paras(0.006, 0.02, 0.0001)
M = gen_paras(1, 10, 1)

labels = read_labels(cf)
max_ari = -1
max_metircs = ()
max_paras = ()
time_cost = 0
data = read_data(sf)
for e in E:
    for m in M:
        start = timeit.default_timer()
        db = dbscan(data, e, m)
        db.process()
        clus = db.get_clusters()
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
        print(max_metircs, max_paras, (ari), (e, m))
print("Result:   ", max_paras, max_metircs, time_cost)
