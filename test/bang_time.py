from gtclu.test.util import gen_paras, read_data, read_labels
import timeit
from pyclustering.cluster.bang import bang
import sys
from sklearn.metrics import rand_score, adjusted_rand_score


sys.setrecursionlimit(50000)


dataset = "30k-20d-10c-0.6"
# L = gen_paras(2, 60, 1)
# M = [0, 5, 10, 50, 100]
para = [(29, 5, 0.6), (30, 100, 0.8), (33, 0, 0.9)]
# for dataset in datasets:
sf = "/home/doors/Code/dataset/efficiency/" + dataset
cf = "/home/doors/Code/dataset/efficiency/" + dataset + "-class"
data = read_data(sf)
labels = read_labels(cf)
max_ri = 0
best_time = 0
# for e in E:
#    width = e / math.sqrt(len(data[0]))
#    sp = int(math.floor(1 / width))
#    for m in M:
# for l in L:
#    for m in M:
for l, m, ari in para:
    start = timeit.default_timer()
    op = bang(data, l, ccore=False, amount_threshould=m)
    op.process()
    clus = op.get_clusters()
    end = timeit.default_timer()

    # prelabels = [-1] * len(data)
    # for i, v in enumerate(clus):
    #    for j in v:
    #        prelabels[j] = i + 1

    # ri = rand_score(labels, prelabels)
    # ari = adjusted_rand_score(labels, prelabels)

    print(ari, end - start)

    # if ri > max_ri:
    #    max_ri = ri
    #    best_time = end - start

# print(dataset, max_ri, best_time, sp, m)
