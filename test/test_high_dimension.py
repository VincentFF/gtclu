from gtclu.gtclu.gtclu_fast import GTCLU
import timeit
import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from gtclu.test.util import read_labels, gen_paras, purity

record = "/home/doors/Code/GTCLU/gtclu/test/record.txt"

dataset = "powersupply"
d = 2
# e = 0.048
# m = 7

E = gen_paras(0.1, 1, 0.1)
M = gen_paras(1, 5, 1)

sf = "/home/doors/Code/dataset/middle/" + dataset
cf = "/home/doors/Code/dataset/middle/" + dataset + "-class"

labels = read_labels(cf)
max_ari = -1
max_metircs = ()
max_paras = ()
time_cost = 0
for e in E:
    for m in M:
        gtclu = GTCLU(e, m, d, algo="tree")
        fp = open(sf)
        start = timeit.default_timer()
        line = fp.readline().strip()
        while line:
            p = np.array(list(map(lambda x: float(x), line.split(","))))
            gtclu.learn_one(p)
            line = fp.readline().strip()
        fp.close()
        gtclu.fit()
        # print(len(gtclu.clusters))
        # print(gtclu.clusters)
        end = timeit.default_timer()

        prelabels = []
        with open(sf) as fp:
            line = fp.readline().strip()
            while line:
                p = np.array(list(map(lambda x: float(x), line.split(","))))
                y = gtclu.predict_one(p)
                prelabels.append(y)
                line = fp.readline().strip()

        ami = adjusted_mutual_info_score(labels, prelabels)
        ari = adjusted_rand_score(labels, prelabels)
        pur = purity(labels, prelabels)

        if ari > max_ari:
            max_ari = ari
            max_metircs = (ari, ami, pur)
            max_paras = (e, m)
            time_cost = end - start
        print(max_metircs, max_paras, (ari), (e, m),
              time_cost, len(gtclu.clusters))

ari, ami, pur = max_metircs
e, m = max_paras
print(ari, ami, pur, e, m, time_cost)
