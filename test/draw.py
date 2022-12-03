from gtclu.gtclu.gtclu import GTCLU
from gtclu.test.util import draw, read_data, read_labels
from sklearn.metrics import adjusted_rand_score
from pyclustering.cluster.optics import optics
from pyclustering.cluster.dbscan import dbscan
from pyclustering.cluster.bang import bang


dataset = "jain"

sf = "/home/doors/Code/dataset/quality/" + dataset
cf = "/home/doors/Code/dataset/quality/" + dataset + "-class"

labels = read_labels(cf)
data = read_data(sf)


# gtclu
e, m = 0.06, 10
d = len(data[0])
gtclu = GTCLU(e, m, d, algo="tree")
for p in data:
    gtclu.learn_one(p)
gtclu.fit()
prelabels = []
for p in data:
    prelabels.append(gtclu.predict_one(p))


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

print(adjusted_rand_score(labels, prelabels))
draw(data, prelabels, "GBSCAN")
