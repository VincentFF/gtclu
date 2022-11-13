from gtclu.gtclu.gtclu import GTCLU
import timeit
import numpy as np

record = "/home/doors/Code/GTCLU/gtclu/test/time-record"

dataset = "100k-15d-10c"
d = 15
e = 0.7
m = 10


sf = "/home/doors/Code/dataset/big/" + dataset

gtclu = GTCLU(e, m, d, algo="tree", tree_level=20)
fp = open(sf)
start = timeit.default_timer()
line = fp.readline().strip()
while line:
    p = np.array(list(map(lambda x: float(x), line.split(","))))
    gtclu.learn_one(p)
    line = fp.readline().strip()
fp.close()
print("start fit")
gtclu.fit()
print(len(gtclu.clusters))
print(len(gtclu.clusters[0]))
end = timeit.default_timer()

print(end - start)
