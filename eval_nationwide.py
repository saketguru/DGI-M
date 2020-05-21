import pickle
import sys
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import gc
from classification import logistic_regression_classification
from mutils import _read_graph_from_edgelist, graph2nx


def fill_set(lines, embedding, X, y, old2new=None, mlb=None,
             location_labels=None, w2v=True):
    for index, line in enumerate(lines):
        if len(line.strip()) == 0:
            continue
        entity, label = line.split("\t")
        label = label.strip()
        if old2new is None:
            X[index:] = embedding[entity:]
            y[index:] = mlb.transform([[location_labels[entity]]])[0]
        else:
            if int(entity) in old2new:
                entity = old2new[int(entity)]
            if w2v:
                entity = str(entity)
            else:
                entity = int(entity)
            X[index][:] = embedding[entity][:]
            y[index][:] = mlb.transform([[label]])[0]

    # print(X.shape, y.shape)


def get_dataset(f, embedding, old2new=None, train_file=None, test_file=None, dimension_len=None, mlb=None,
                location_labels=None, w2v=True):
    test_lines = open(foldername + test_file).readlines()
    train_lines = open(foldername + train_file).readlines()
    X_train, X_test, y_train, y_test = np.zeros((len(train_lines), 128)), np.zeros((len(test_lines), 128)), np.zeros(
        (len(train_lines), dimension_len)), np.zeros((len(test_lines), dimension_len))

    fill_set(train_lines, embedding, X_train, y_train, old2new, mlb, location_labels, w2v)
    fill_set(test_lines, embedding, X_test, y_test, old2new, mlb, location_labels, w2v)

    return X_train, X_test, y_train, y_test


def evaluate():
    embedding = np.load(features_file)
    embedding = embedding.reshape((-1, 128))

    for key, val in variants.items():
        location_labels, location_labels_counts, mlb, dimension_len = variants_res[key]
        fw = open(result_folder + "result_all_dgi_%s_%s_%s.txt" % (dataset, param, key), "w+")
        for tsz in range(1, 10):
            micros = []
            macros = []
            for fold in range(folds):
                X_train, X_test, y_train, y_test = get_dataset(fold, embedding, mapping.old2new,
                                                               train_file % (dataset, key, fold, tsz / 10.0),
                                                               test_file % (dataset, key, fold, tsz / 10.0),
                                                               dimension_len, mlb, location_labels, w2v=False)
                ros = RandomOverSampler()
                X_train, y_train = ros.fit_sample(X_train, y_train)
                lf_classifer, results = logistic_regression_classification(X_train, X_test, y_train,
                                                                           y_test,
                                                                           None, True)
                micros.append(results['micro'])
                macros.append(results['macro'])

            print("DGI: %s,%s,%s,%s,%s" % (dataset, param, tsz / 10.0, np.mean(micros), np.mean(macros)))
            fw.write("DGI: %s,%s,%s,%s,%s\n" % (dataset, param, tsz / 10.0, np.mean(micros), np.mean(macros)))
        fw.close()
        gc.collect()


class Ctrl:
    pass


ctrl = Ctrl()
ctrl.debug_mode = False

if __name__ == '__main__':
    features_file = sys.argv[1]
    # dataset = "columbus"
    # foldername = "gps_drivers/"
    # result_folder = "result_files/"

    dataset = "philadelphia"
    foldername = "data_files/"
    result_folder = "results/"

    param = features_file.replace(".emb.npy", "").replace(dataset + "_", "")
    variants_res = pickle.load(open(foldername + "variant_res_%s.pkl" % dataset, "rb"))
    variants = dict()
    folds = 5
    variants[3] = ['fast_food', 'service', 'restaurant', 'school', 'place_of_worship', 'motorway',
                   'bank', 'hotel', 'supermarket', 'parking', 'convenience', 'trunk', 'commercial', 'cafe',
                   'car_repair', 'apartments', 'university']

    variants[4] = ['fast_food', 'service', 'restaurant', 'school', 'place_of_worship', 'motorway',
                   'bank', 'hotel', 'supermarket', 'parking', ]

    variants[5] = ['place_of_worship', 'motorway', 'yes', 'bank', 'hotel', 'supermarket', 'parking', 'convenience',
                   'trunk']
    # train_file = "train_selected_class%s_%s_tp_%s.txt"
    # test_file = "test_selected_class%s_%s_tp_%s.txt"
    train_file = "train_%s_%s_%s_tp_%s.txt"
    test_file = "test_%s_%s_%s_tp_%s.txt"

    graph, mapping = _read_graph_from_edgelist(ctrl, "%s_graph.csv" % dataset)
    if mapping is None:
        class Mapping:
            old2new = {}


        mapping = Mapping()
        for i in range(graph.node_num):
            mapping.old2new[i] = i
    evaluate()
