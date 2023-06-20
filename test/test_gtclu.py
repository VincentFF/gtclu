from gtclu.gtclu.gtclu import GTCLU
import timeit
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from gtclu.test.util import read_labels, gen_paras, read_data, purity


datasets = [
    ## "Compound",
    ## "D31",
    ## "flame",
    # "iris",
    ## "pathbased",
    # "R15",
    "g2-2-30",
    # "g2-4-30",
    # "g2-8-30",
    # "s1",
    # "s2",
    # "s3",
    # "s4",
]

E = gen_paras(0.01, 0.2, 0.005)
M = gen_paras(1, 50, 0.5)
for dataset in datasets:

    sf = "/home/doors/Code/dataset/quality/" + dataset
    cf = "/home/doors/Code/dataset/quality/" + dataset + "-class"

    labels = read_labels(cf)
    max_ari = -1
    max_metircs = ()
    max_paras = ()
    time_cost = 0
    data = read_data(sf)
    d = len(data[0])
    for e in E:
        for m in M:

            start = timeit.default_timer()
            gtclu = GTCLU(e, m, d, algo="bfs")
            for p in data:
                gtclu.learn_one(p)
            gtclu.fit()
            end = timeit.default_timer()

            prelabels = []
            for p in data:
                prelabels.append(gtclu.predict_one(p))
            ami = adjusted_mutual_info_score(labels, prelabels)
            ari = adjusted_rand_score(labels, prelabels)
            pur = purity(labels, prelabels)

            if ari > max_ari:
                max_ari = ari
                max_metircs = (ari, ami, pur)
                max_paras = (e, m)
                time_cost = end - start
            print(max_metircs, max_paras)
    print(
        dataset,
        max_metircs[0],
        max_metircs[1],
        max_metircs[2],
        max_paras[0],
        max_paras[1],
    )
