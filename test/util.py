from cProfile import label
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def gen_paras(a, b, acc):
    res = []
    if acc > 0:
        while a <= b:
            res.append(a)
            a += acc
    else:
        while a >= b:
            res.append(a)
            a += acc
    return res


def read_labels(path):
    labels = []
    with open(path) as fp:
        labels = map(lambda x: int(x.strip()), fp.readlines())
    labels = list(labels)
    return labels


def read_data(path):
    data = []
    with open(path) as fp:
        str_data = fp.readlines()
        for line in str_data:
            tem = line.strip().split(",")
            data.append(list(map(lambda x: float(x), tem)))
    return np.array(data)


def purity(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)


# def draw(data, labels, name=None):
#    scatter_x = data[:, 0]
#    scatter_y = data[:, 1]
#    _, ax = plt.subplots()
#    for g in np.unique(labels):
#        ix = np.where(labels == g)
#        if g == -1:
#            ax.scatter(scatter_x[ix], scatter_y[ix], label=g, s=5, color="gray")
#        else:
#            ax.scatter(scatter_x[ix], scatter_y[ix], s=5, label=g)
#    # ax.legend()
#    title = "pic"
#    if name:
#        # ax.set_title(name)
#        title = name
#    plt.xticks([])
#    plt.yticks([])

#    plt.savefig(title + ".eps", dpi=600)
#    plt.show()
colors = ["#1862d9", "#d91862", "#097533"]
# colors = [
#    "#377eb8",
#    "#ff7f00",
#    "#4daf4a",
#    "#f781bf",
#    "#a65628",
#    "#984ea3",
#    "#999999",
#    "#e41a1c",
#    "#dede00",
#    "#595959",
#    "#800080",
#    "#FFFF00",
#    "#00FFFF",
#    "#008080",
#    "#800000",
#    "#00FF00",
#    "#008000",
#    "#000080",
#    "#FF00FF",
#    "#0000FF",
#    "#7F7F7F",
#    "#FF0000",
#    "#FFA500",
#    "#9ACD32",
#    "#A52A2A",
#    "#7FFF00",
#    "#D2691E",
#    "#DC143C",
#    "#006400",
#    "#2E8B57",
#    "#ADFF2F",
#    "#8B0000",
#    "#556B2F",
#    "#FF8C00",
#    "#9932CC",
#    "#48D1CC",
#    "#FF1493",
#    "#1E90FF",
#    "#228B22",
#    "#FFD700",
# ]
custom_cmap = ListedColormap(colors)


def draw(data, labels, names=None):
    if not isinstance(labels[0], list):
        labels = [labels]
    scatter_x = data[:, 0]
    scatter_y = data[:, 1]
    fig, ax = plt.subplots(1, len(labels))
    fig.set_size_inches(10, 5)
    for i in range(len(labels)):
        # cmp = plt.get_cmap("CMRmap", len(set(labels[i])))
        # for g in np.unique(labels[i]):
        #    ix = np.where(labels[i] == g)
        #    # if g == -1:
        #    #    ax[i].scatter(scatter_x[ix], scatter_y[ix], label=g, s=5, c="gray")
        #    # else:
        #    ax[i].scatter(scatter_x[ix], scatter_y[ix], label=g, s=5)
        ax[i].scatter(scatter_x, scatter_y, c=labels[i], s=5, cmap=custom_cmap)
        # ax[i].legend()
        if names:
            ax[i].set_title(names[i])
    fig.savefig("pic.png", dpi=600)
    plt.show()
