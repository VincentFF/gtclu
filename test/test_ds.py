#from util import gen_paras, read_labels, read_data, purity, draw
import timeit
#from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
#from sgdbscan.test.dbscan import DBSCAN
# from sklearn.cluster import DBSCAN
from pyclustering.cluster.dbscan import dbscan
from util import read_data
import sys
sys.setrecursionlimit(10000)

#datasets = ["skin"]
record = "/home/doors/Code/stream/sgdbscan/test/db-record"

dataset = "1m-30d-10c"
e = 0.5
m = 20

# for dataset in datasets:

sf = "/home/doors/Code/stream/dataset/big/"+dataset
#cf = "/home/doors/Code/stream/dataset/middle/"+dataset+"-class"

#epsilon = gen_paras(0.04, 0.06, 0.001)
#minpts = gen_paras(2, 10, 1)
#minpts = [5]

#labels = read_labels(cf)

#max_ari = -1
#max_paras = []
#max_scores = []

##best_prelabels = []
# for e in epsilon:
#    for m in minpts:
start = timeit.default_timer()
data = read_data(sf)
db = dbscan(data, e, m, ccore=False)
db.process()
#clus = db.get_clusters()
end = timeit.default_timer()
print(end-start)
#prelabels = [0]*len(data)

# for i, v in enumerate(clus):
#    for j in v:
#        prelabels[j] = i+1

#purity_score = purity(labels, prelabels)
#ari = adjusted_rand_score(labels, prelabels)
#ami = adjusted_mutual_info_score(labels, prelabels)
# if ari > max_ari:
#    max_ari = ari
#    max_paras = [e, m, end-start]
#    max_scores = [purity_score, ari, ami]
#    best_prelabels = prelabels
# print(dataset)
# print(max_paras)
# print(max_scores)
# with open(record, "a") as f:
#    f.write(dataset+"\n")
#    f.write(",".join(map(str, max_paras))+"\n")
#    f.write(",".join(map(str, max_scores))+"\n")
#    f.write("\n")
#    f.write("\n")

#draw(data, best_prelabels)
