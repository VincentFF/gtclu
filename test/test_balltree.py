from gtclu.gtclu.gtclu import GTCLU
import timeit
import numpy as np

record = "/home/doors/Code/GTCLU/gtclu/test/time-record"

dataset = "50k-20d-10c"
d = 20
e = 0.5
m = 5


sf = "/home/doors/Code/dataset/big/" + dataset

gtclu = GTCLU(e, m, d, algo="tree")
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
end = timeit.default_timer()

print(end - start)
