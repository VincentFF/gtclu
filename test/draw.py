from gtclu.gtclu.gtclu import GTCLU
from gtclu.test.util import draw, read_data, read_labels, gen_paras
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# from pyclustering.cluster.optics import optics
# from pyclustering.cluster.dbscan import dbscan
# from pyclustering.cluster.bang import bang


dataset = "flame"
e, m = 0.11, 7.9
sf = "/home/doors/Code/dataset/quality/" + dataset
cf = "/home/doors/Code/dataset/quality/" + dataset + "-class"

labels = read_labels(cf)
data = read_data(sf)

E = gen_paras(0.03, 0.12, 0.01)
M = gen_paras(1, 6, 0.1)

d = len(data[0])
gtclu = GTCLU(e, m, d, algo="tree")
for p in data:
    gtclu.learn_one(p)
gtclu.fit()
prelabels = []
for p in data:
    prelabels.append(gtclu.predict_one(p))

print(
    adjusted_rand_score(labels, prelabels),
    adjusted_mutual_info_score(labels, prelabels),
)
draw(data, [labels, prelabels], ["Ground Truth", "GTCLU"])


# d = len(data[0])
# max_ari = 0
# for e in E:
#    for m in M:
#        gtclu = GTCLU(e, m, d, algo="tree")
#        for p in data:
#            gtclu.learn_one(p)
#        gtclu.fit()
#        prelabels = []
#        for p in data:
#            prelabels.append(gtclu.predict_one(p))
#        ari = adjusted_rand_score(labels, prelabels)
#        if ari > max_ari:
#            max_ari = ari
#            max_e, max_m = e, m
#        # print(adjusted_rand_score(labels, prelabels), e, m)
#        print("max:", max_ari, max_e, max_m)

# optics
# e, m = (0.123, 10)
# op = optics(data, e, m, ccore=False)
# op.process()
# clus = op.get_clusters()
# prelabels = [-1] * len(data)

# for i, v in enumerate(clus):
#    for j in v:
#        prelabels[j] = i + 1


# DBSCAN
# e, m = (0.123, 10)
# op = optics(data, e, m, ccore=False)
# op.process()
# clus = op.get_clusters()
# prelabels = [-1] * len(data)
# for i, v in enumerate(clus):
#    for j in v:
#        prelabels[j] = i + 1

## kmeans
# kmeans = KMeans(n_clusters=3).fit(data)
# prelabels = kmeans.labels_
