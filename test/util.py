from cProfile import label
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
import matplotlib.pyplot as plt


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


def draw(data, labels, name=None):
    scatter_x = data[:, 0]
    scatter_y = data[:, 1]
    _, ax = plt.subplots()
    for g in np.unique(labels):
        ix = np.where(labels == g)
        # if g == -1:
        #    ax.scatter(scatter_x[ix], scatter_y[ix], label=g, s=5, color="gray")
        # else:
        ax.scatter(scatter_x[ix], scatter_y[ix], s=5, label=g)
    # ax.legend()
    if name:
        ax.set_title(name)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# def draw(data, labels,names=None):
#    if not isinstance(labels[0],list):
#        labels = [labels]
#    scatter_x = data[:,0]
#    scatter_y = data[:,1]
#    fig,ax = plt.subplots(1,len(labels))
#    for i in range(len(labels)):
#        for g in np.unique(labels[i]):
#            ix = np.where(labels[i]==g)
#            if g == -1:
#                ax[i].scatter(scatter_x[ix],scatter_y[ix],label=g, s=5,c="black",marker='*')
#            else:
#                ax[i].scatter(scatter_x[ix],scatter_y[ix],label=g, s=5)
#        ax[i].legend()
#        if names:
#            ax[i].set_title(names[i])
#    plt.show()
