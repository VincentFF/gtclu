from gtclu.gtclu.gtclu_kdtree import GTCLU
from gtclu.test.util import gen_paras, read_data, read_labels
import timeit
import sys
import math

# from sklearn.metrics import rand_score, adjusted_rand_score

sys.setrecursionlimit(50000)


D = 20
dataset = "20k-20d-20c-0.6"
E = gen_paras(2, 10, 1)
# M = gen_paras(2, 10, 1)
M = [1, 5, 20, 50]
sf = "/home/doors/Code/dataset/efficiency/" + dataset
# cf = "/home/doors/Code/dataset/efficiency/" + dataset + "-class"
data = read_data(sf)
# labels = read_labels(cf)
max_ri = 0
best_time = 0
# for e in E:
#    for m in M:
# for e, m, ari in para:
for m in M:
    for e in E:
        ep = math.sqrt(D) / e
        # now_time = 0
        # for i in range(10):
        start = timeit.default_timer()
        gtclu = GTCLU(ep, m, D, algo="tree")
        for p in data:
            gtclu.learn_one(p)
        gtclu.fit()
        end = timeit.default_timer()
        # now_time += end - start
        print(e, m, end - start)

    # prelabels = []
    # for p in data:
    #    prelabels.append(gtclu.predict_one(p))

    # ri = rand_score(labels, prelabels)
    # ari = adjusted_rand_score(labels, prelabels)
    # print(ari, ri, end - start, e, m)

#        if ri > max_ri:
#            max_ri = ri
#            best_time = end - start

# print(dataset, max_ri, best_time, e, m)
