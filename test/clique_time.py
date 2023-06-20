from gtclu.test.util import gen_paras, read_data, read_labels
import timeit
from pyclustering.cluster.clique import clique, clique_visualizer
import sys
from sklearn.metrics import rand_score, adjusted_rand_score


sys.setrecursionlimit(50000)


dataset = "100k-3d-10c-0.6"
E = gen_paras(10, 50, 5)
M = [0, 5, 50, 100]
# for dataset in datasets:
# para = [(2, 0, 1)]
sf = "/home/doors/Code/dataset/efficiency/" + dataset
cf = "/home/doors/Code/dataset/efficiency/" + dataset + "-class"
data = read_data(sf)
labels = read_labels(cf)
# max_ri = 0
# best_time = 0
# for e in E:
#    width = e / math.sqrt(len(data[0]))
#    sp = int(math.floor(1 / width))
#    for m in M:
for sp in E:
    for m in M:
        # for sp, m, ari in para:
        start = timeit.default_timer()
        op = clique(data, sp, m, ccore=False)
        op.process()
        clus = op.get_clusters()
        end = timeit.default_timer()
        # show cells
        # clique_visualizer.show_grid(op.get_cells(), data)
        # clique_visualizer.show_clusters(data, clus, op.get_noise())

        prelabels = [-1] * len(data)
        for i, v in enumerate(clus):
            for j in v:
                prelabels[j] = i + 1

        # ri = rand_score(labels, prelabels)
        ari = adjusted_rand_score(labels, prelabels)
        print(sp, m, end - start, ari)

    # print(ari, ri, end - start, sp, m)

    # if ri > max_ri:
    #    max_ri = ri
    #    best_time = end - start
