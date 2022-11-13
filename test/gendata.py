from sklearn.datasets import make_blobs

name = "100k-30d-5c"
forder = "/home/doors/Code/dataset/origin"

file = forder + "/" + name
label = "/home/doors/Code/dataset/big/" + name + "-class"


X, y = make_blobs(
    n_samples=100000, centers=5, n_features=30, shuffle=True, random_state=0
)

with open(label, "w") as f:
    for i in y:
        f.write(str(i) + "\n")

with open(file, "w") as f:
    for data in X:
        line = ",".join(map(str, data))
        f.write(line + "\n")
