from sgdbscan.cluster.gbscan_single import GBSCAN
import timeit
import numpy as np

record = "/home/doors/Code/stream/sgdbscan/test/gbscan-record"

dataset = "1m-30d-10c"
d = 30
e = 2
m = 30


sf = "/home/doors/Code/stream/dataset/big/"+dataset
# cf = "/home/doors/Code/stream/dataset/middle/"+dataset+"-class"

gbscan = GBSCAN(e, m, d, algo="tree", tree_level=20)
fp = open(sf)
start = timeit.default_timer()
line = fp.readline().strip()
while line:
    p = np.array(list(map(lambda x: float(x), line.split(','))))
    gbscan.learn_one(p)
    line = fp.readline().strip()
fp.close()
print("start fit")
gbscan.fit()
end = timeit.default_timer()

print(end-start)
