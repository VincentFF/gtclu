from sklearn.datasets import make_blobs

name = "10k-10d-10c-0.6"
forder = "/home/doors/Code/dataset/origin"

file = forder + "/" + name
label = "/home/doors/Code/dataset/big/" + name + "-class"


X, y = make_blobs(
    n_samples=10000, centers=10, n_features=10, shuffle=True, random_state=0,cluster_std=0.6
)

with open(label, "w") as f:
    for i in y:
        f.write(str(i) + "\n")

with open(file, "w") as f:
    for data in X:
        line = ",".join(map(str, data))
        f.write(line + "\n")
