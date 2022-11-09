from sgdbscan.cluster.gbscan_single import GBSCAN
import timeit
import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sgdbscan.test.util import read_labels, gen_paras

record = "/home/doors/Code/stream/sgdbscan/test/gbscan-record"

dataset = "1m-30d-10c"
d = 30
# e = 0.048
# m = 7

E = gen_paras(0.5, 0.5, 1)
M = gen_paras(20, 20, 1)

sf = "/home/doors/Code/stream/dataset/big/"+dataset
cf = "/home/doors/Code/stream/dataset/big/"+dataset+"-class"

labels = read_labels(cf)
max_ari = -1
for e in E:
    for m in M:
        gbscan = GBSCAN(e, m, d, algo="tree", tree_level=15)
        fp = open(sf)
        start = timeit.default_timer()
        line = fp.readline().strip()
        while line:
            p = np.array(list(map(lambda x: float(x), line.split(','))))
            gbscan.learn_one(p)
            line = fp.readline().strip()
        fp.close()
        gbscan.fit()
        print(len(gbscan.clusters))
        print(gbscan.clusters)
        end = timeit.default_timer()

        prelabels = []
        with open(sf) as fp:
            line = fp.readline().strip()
            while line:
                p = np.array(
                    list(map(lambda x: float(x), line.split(','))))
                y = gbscan.predict_one(p)
                prelabels.append(y)
                line = fp.readline().strip()

        ami = adjusted_mutual_info_score(labels, prelabels)
        ari = adjusted_rand_score(labels, prelabels)

        if ari > max_ari:
            max_ari = ari
            max_metircs = (ari, ami)
            max_paras = (e, m)
            time_cost = end-start
        print(max_metircs, max_paras, (ari, ami), (e, m))
print("Result:   ", max_paras, max_metircs, time_cost)
