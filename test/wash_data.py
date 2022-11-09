

sf = "/home/doors/Code/stream/dataset/origin/1m-30d-10c"
df = "/home/doors/Code/stream/dataset/big/1m-30d-10c"
dc = "/home/doors/Code/stream/dataset/middle/sensor-class"

d = 30
min_max = [[float("inf"), float("-inf")] for i in range(d)]

sd = []
cd = []

with open(sf) as fp:
    line = fp.readline().strip()
    while line:
        data = list(map(float, line.split(",")))
        # cd.append(data[-1])
        sd.append(data[:d])
        for i in range(d):
            min_max[i][0] = min(min_max[i][0], float(data[i]))
            min_max[i][1] = max(min_max[i][1], float(data[i]))
        line = fp.readline().strip()

# with open(df, "w") as fp1, open(dc, "w") as fp2:
#     for i in range(len(sd)):
#         for j in range(d):
#             sd[i][j] = (sd[i][j] - min_max[j][0]) / \
#                 (min_max[j][1] - min_max[j][0])
#         fp1.write(",".join(list(map(str, sd[i])))+"\n")
#         fp2.write(str(int(cd[i])) + "\n")

with open(df, "w") as fp1:
    for i in range(len(sd)):
        for j in range(d):
            sd[i][j] = (sd[i][j] - min_max[j][0]) / \
                (min_max[j][1] - min_max[j][0]
                 ) if min_max[j][1] != min_max[j][0] else 0
        fp1.write(",".join(list(map(str, sd[i])))+"\n")
