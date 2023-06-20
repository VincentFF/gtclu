log = "gtclu-ari-30k-20d.log"
logto = log + ".to"

with open(log) as fl:
    data = list(map(lambda x: x.split(), fl.readlines()))

fdata = []
for v in data:
    fdata.append(list(map(float, v)))

sdata = sorted(fdata, key=lambda x: x[2])

with open(logto, "w") as fl:
    for v in sdata:
        fl.write(" ".join(map(str, v)) + "\n")
