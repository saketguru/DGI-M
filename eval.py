import sys
import numpy as np
from gensim.models import KeyedVectors
import csv
import ast
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler
import geopy.distance
from sklearn.preprocessing import MultiLabelBinarizer
from classification import logistic_regression_classification
from sklearn.utils import shuffle
import pickle
from mutils import read_graph


def load_embeddings(fname, mappings, info):
    model = KeyedVectors.load_word2vec_format(fname)
    new2old_fp = open(mappings)
    new2old = dict()
    for line in new2old_fp.readlines():
        eles = line.decode("utf-8-sig").encode("utf-8").strip().split("\t")
        new2old[eles[0]] = eles[1]

    loc_id2name = dict()
    with open(info) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            loc_id2name[row[-3]] = row[-2]

    print(loc_id2name[new2old['17']])

    if '17' in model:
        results = model.similar_by_word('17', topn=10)
        for result in results:
            print(loc_id2name[new2old[result[0]]])


def get_locs(fname):
    locs = dict()
    with open(fname) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        headers = next(reader, None)

        for row in reader:
            pid = row[0]
            lat = row[5]
            lon = row[4]
            label = row[-4].strip().split(":")[-1].strip()
            if len(label) < 2:
                continue
            if pid not in locs:
                locs[pid] = []
            locs[pid].append((lat, lon, label))
    return locs


def create_labels(info, dist_th):
    labels = dict()
    labels_dist = dict()
    locs = get_locs("ahdc_data/ylocations_process070918.csv")
    # print (locs)
    unknown = "UNKNOWN_PLACE"

    counts = 0
    unknown_entries = 0
    all_labels = []
    with open(info) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')

        for row in reader:
            if len(row[-1].strip()) < 1:
                continue
            osm = ast.literal_eval(row[-1])
            # print (osm)
            pid = str(row[1])
            lat = row[6]
            lon = row[7]

            # if pid != '1809':
            #     continue

            if pid not in locs:
                if osm['osm_id'] not in labels:
                    labels[osm['osm_id']] = unknown
            else:
                new_label = unknown
                min_dist = 10000
                # print("\n ---- OSM ID %s lat %s lon %s \n" % (osm['osm_id'], lat, lon,))
                for entries in locs[pid]:
                    reg_lat, reg_lon, reg_label = entries
                    dist = geopy.distance.vincenty((reg_lon, reg_lat), (lon, lat)).meters
                    # print(dist, reg_label)

                    if dist < min_dist and dist <= dist_th:
                        new_label = reg_label
                        min_dist = dist

                if new_label != unknown:
                    unknown_entries += 1

                if new_label == unknown:
                    continue

                all_labels.append(new_label)

                if osm['osm_id'] not in labels:
                    labels[osm['osm_id']] = [new_label]  # [(new_label, min_dist, pid)]
                else:
                    labels[osm['osm_id']].append(new_label)

                # elif labels[osm['osm_id']][0] == unknown and new_label != unknown:
                #     labels[osm['osm_id']] = [new_label]  # [(new_label, min_dist, pid)]
                # elif labels[osm['osm_id']][0] != new_label and new_label != unknown:
                #     print("lat %s lon %s %s %s %s" % (lat, lon, labels[osm['osm_id']], (new_label, min_dist), pid))
                # labels[osm['osm_id']].append(new_label)
                # counts += 1

    print("Overlap %s, known entries %s" % (counts, unknown_entries))

    label_counts = Counter(all_labels)
    print(label_counts)

    min = 50

    remove = []
    for key, val in label_counts.items():
        if val < min:
            remove.append(key)

    all_labels = set(all_labels)
    for r_lbl in remove:
        all_labels.remove(r_lbl)

    print(all_labels)
    mlb = MultiLabelBinarizer()
    mlb.fit([set(all_labels)])
    print(mlb.classes_)

    fp = open("Multi_labels_min_threshold_50_nominatim.txt", mode="w+")
    for key, val in labels.items():
        if val == [unknown]:
            continue

        nvals = []
        for v in val:
            if v not in remove:
                nvals.append(v)

        val = nvals
        if len(val) == 0:
            continue

        val = set(val)
        # print (list(mlb.transform([val])[0]))
        fp.write(str(key) + "\t" + str(list(mlb.transform([val])[0])) + "\n")

    fp.close()

    # le = LabelEncoder()
    # le.fit(labels.values())
    # fp = open("Labels.txt", mode="w+")
    # for key, val in labels.items():
    #     # print(key, val)
    #     # print(le.transform([val]))
    #     fp.write(str(key) + "\t" + str(le.transform([val])) + "\n")
    # fp.close()
    #
    # print(Counter(labels.values()))
    # fp = open("Label_classes.txt", mode="w+")
    # fp.write(" ".join(list(le.classes_)))
    # fp.close()


def create_features_and_labels(features_file, old2new, labels_file):
    features, labels = dict(), dict()
    features = np.load(features_file)
    features = features.reshape((-1, features.shape[2]))

    for line in open(labels_file).readlines():
        records = line.split("\t")
        labels[old2new[int(records[0])]] = ast.literal_eval(records[1].strip())

    y_dim = len(labels[list(labels.keys())[0]])
    X = np.zeros((len(labels.keys()), 128))
    y = np.zeros((len(labels.keys()), y_dim))

    for i, key in enumerate(list(labels.keys())):
        X[i][:] = features[key]
        y[i][:] = np.array(labels[key])

    return X, y


def compute_microf1(X, y):
    test_percents = range(1, 10, 1)
    print("Optimizing based on Micro")

    print("Test Percent, Micro-f1, Macro-f1, Weighted\n")

    for test_percent in test_percents:

        skf = StratifiedShuffleSplit(n_splits=3, test_size=test_percent / 10.0, random_state=42)
        # X_train, X_test, y_train_, y_test_ = train_test_split(X, y, test_size=test_percent / 10.0)
        result = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            param_grid = [{'C': [1, 10, 100], 'gamma': [1, 0.1, 0.01]}]

            clf = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5, scoring='f1_micro', n_jobs=-1, verbose=0)
            ros = RandomOverSampler()
            X_train, y_train = ros.fit_sample(X_train, y_train)

            clf.fit(X_train, y_train)

            y_predict = clf.predict(X_test)
            result.append((f1_score(y_test, y_predict, average='micro'), f1_score(y_test, y_predict, average='macro'),
                           f1_score(y_test, y_predict, average='weighted')))

        print("%s, %s, %s, %s" % (
            test_percent / 10.0,
            (result[0][0] + result[1][0] + result[2][0] / 3.0),
            (result[0][1] + result[1][1] + result[2][1] / 3.0),
            (result[0][2] + result[1][2] + result[2][2] / 3.0)))


def store_train_test_sets(X, y):
    test_percents = range(1, 10, 1)
    num_shuffles = 5

    labels_count = y.shape[1]
    multi_label_binarizer = MultiLabelBinarizer(range(labels_count))
    multi_label_binarizer.fit(y)

    train_test_sets = dict()

    print("Optimizing based on Micro")
    print("Test Percent, Micro-f1, Macro-f1, Weighted\n")

    for test_percent in test_percents:
        train_test_sets[str(test_percent)] = dict()
        for i in range(num_shuffles):
            skf = shuffle(X, y)
            X, y = skf
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent / 10.0)
            train_test_sets[str(test_percent)][i] = (X_train, X_test, y_train, y_test)

    f = open("ahdc_train_test.pkl", "wb")
    pickle.dump(train_test_sets, f)
    f.close()


def compute_microf1_multi_label(X, y, grid_search):
    test_percents = range(1, 10, 1)
    num_shuffles = 5

    labels_count = y.shape[1]
    multi_label_binarizer = MultiLabelBinarizer(range(labels_count))
    multi_label_binarizer.fit(y)

    final_results = dict()

    print("Optimizing based on Micro")
    print("Test Percent, Micro-f1, Macro-f1, Weighted\n")

    for test_percent in test_percents:
        result = []
        for i in range(num_shuffles):
            skf = shuffle(X, y)
            X, y = skf
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent / 10.0)

            clf, results = logistic_regression_classification(X_train, X_test, y_train, y_test, multi_label_binarizer,
                                                              grid_search)
            result.append((results['micro'], results['macro']))

        micro_avg = 0.0
        macro_avg = 0.0
        test_percent = float(test_percent)
        for i in range(num_shuffles):
            micro_avg += result[i][0]
            macro_avg += result[i][1]

        print("%s, %s, %s" % (test_percent / 10.0, micro_avg / num_shuffles, macro_avg / num_shuffles))
        final_results[test_percent] = (micro_avg / num_shuffles, macro_avg / num_shuffles)

    return final_results


if __name__ == '__main__':
    dataset = sys.argv[1]
    embeds = np.load(dataset)
    from scripts.create_graph import Ctrl

    ctrl = Ctrl()
    ctrl.debug_mode = False
    graph, mapping = read_graph(ctrl, "train_nominatim_graph_0.txt")
    labels = "Multi_labels_min_threshold_50_nominatim.txt"
    X, y = create_features_and_labels(dataset, mapping.old2new, labels)
    final_results = compute_microf1_multi_label(X, y, True)
    fww = open("result_%s.txt" % (dataset.replace(".emb.npy", "")), "w+")
    for key, vals in final_results.items():
        fww.write("DGI\t%s\t%s\t%s\t\n" % (key / 10.0, vals[0], vals[1]))
        print("DGI\t%s\t%s\t%s\t\n" % (key / 10.0, vals[0], vals[1]))
        fww.flush()
