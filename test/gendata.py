from sklearn.datasets import make_blobs

name = "300k-30d-10c"
forder = "/home/doors/Code/dataset/origin"

file = forder + "/" + name
label = "/home/doors/Code/dataset/big/" + name + "-class"


X, y = make_blobs(
    n_samples=300000, centers=10, n_features=30, shuffle=True, random_state=0
)

with open(label, "w") as f:
    for i in y:
        f.write(str(i) + "\n")

with open(file, "w") as f:
    for data in X:
        line = ",".join(map(str, data))
        f.write(line + "\n")
