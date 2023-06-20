from gtclu.test.util import gen_paras, read_data, read_labels
import timeit
from pyclustering.cluster.dbscan import dbscan
import sys
from sklearn.metrics import rand_score, adjusted_rand_score

sys.setrecursionlimit(50000)


# datasets = [
#    "30k-3d-10c-0.6",
# ]

dataset = "50k-3d-20c-0.6"
E = gen_paras(0.02, 0.1, 0.02)
# M = gen_paras(5, 50, 100)
M = [50, 100]
# para = [(0.115, 6, 0.8), (0.13, 61, 0.9)]
# E = gen_paras(0.05, 0.5, 0.005)
# M = gen_paras(1, 100, 5)
# for dataset in datasets:
sf = "/home/doors/Code/dataset/efficiency/" + dataset
cf = "/home/doors/Code/dataset/efficiency/" + dataset + "-class"
data = read_data(sf)
labels = read_labels(cf)
max_ri = 0
best_time = 0
# for e in E:
#    for m in M:
# for e, m, ari in para:
for e in E:
    for m in M:
        start = timeit.default_timer()
        db = dbscan(data, e, m, ccore=False)
        db.process()
        clus = db.get_clusters()
        end = timeit.default_timer()
        print(e, m, end - start)

#        prelabels = [-1] * len(data)
#        for i, v in enumerate(clus):
#            for j in v:
#                prelabels[j] = i + 1

#        ri = rand_score(labels, prelabels)
#        ari = adjusted_rand_score(labels, prelabels)

#        print(ari, ri, end - start, e, m)

#        if ri > max_ri:
#            max_ri = ri
#            best_time = end - start

# print(dataset, max_ri, best_time, e, m)
