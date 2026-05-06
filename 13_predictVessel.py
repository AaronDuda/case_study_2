import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import functools
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from kneed import KneeLocator

def hh_mm_ss2seconds(hh_mm_ss):
    return functools.reduce(lambda acc, x: acc*60 + x, map(int, hh_mm_ss.split(':')))

def load_and_preprocess(csv_path):
    # load data and convert hh:mm:ss to seconds
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})

    # select features 
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()

    # Standardization 
    return preprocessing.StandardScaler().fit(X).transform(X)

def load_vid_labels(csv_path):
    # load data
    df = pd.read_csv(csv_path)

    # select VID's
    return df['VID'].to_numpy()




def predictor_baseline(csv_path): 
    X = load_and_preprocess(csv_path)

    # k-means with K = number of unique VIDs of set1
    K = 20 
    model = KMeans(n_clusters=K, random_state=123, n_init='auto').fit(X)

    # predict cluster numbers of each sample
    labels_pred = model.predict(X)
    return labels_pred


def get_baseline_score():
    file_names = ['set1.csv', 'set2.csv']
    for file_name in file_names:
        csv_path = './Data/' + file_name
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
        labels_pred = predictor_baseline(csv_path)
        rand_index_score = adjusted_rand_score(labels_true, labels_pred)
        print(f'Adjusted Rand Index Baseline Score of {file_name}: {rand_index_score:.4f}')


def evaluate():
    csv_path = './Data/set2.csv'
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
    labels_pred = predictor(csv_path)
    rand_index_score = adjusted_rand_score(labels_true, labels_pred)
    print(f'Adjusted Rand Index Score of set2.csv: {rand_index_score:.4f}')


def predictor(csv_path):
    all_cluster_labels = []
    ks = []
    inertias = []
    sil_scores = []
    X = load_and_preprocess(csv_path)
    for n_clusters in range(2, 100):
        clusterer = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=123)
        cluster_labels = clusterer.fit_predict(X)

        all_cluster_labels.append(cluster_labels)
        ks.append(n_clusters)
        inertias.append(clusterer.inertia_)
        sil_scores.append(silhouette_score(X, cluster_labels))

    ks = np.array(ks)
    sil_scores = np.array(sil_scores)

    # elbow method on inertia
    knee = KneeLocator(list(map(int, ks)), inertias, curve='convex', direction='decreasing')
    elbow_k = knee.knee if knee.knee is not None else ks[np.argmax(sil_scores)]

    # pick K closest to elbow that also maximizes silhouette within a window around the elbow
    window = 2
    elbow_idx = np.where(ks == elbow_k)[0][0]
    lo = max(0, elbow_idx - window)
    hi = min(len(ks), elbow_idx + window + 1)
    best_index = lo + np.argmax(sil_scores[lo:hi])

    print(f"Elbow K = {elbow_k}, Predicted K = {int(ks[best_index])}")

    return all_cluster_labels[best_index]


if __name__=="__main__":
    get_baseline_score()
    evaluate()


