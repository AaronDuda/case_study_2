import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans, HDBSCAN
import functools
from sklearn.metrics.cluster import adjusted_rand_score

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

def load_and_preprocess_predictor(csv_path):
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    X = np.column_stack((
        df['SEQUENCE_DTTM'].to_numpy(dtype=float),
        df['LAT'].to_numpy(dtype=float),
        df['LON'].to_numpy(dtype=float),
    ))
    X = preprocessing.StandardScaler().fit(X).transform(X)
    return X * np.array([0.5, 3.0, 3.0])

def renumber_labels(labels):
    label_map = {old_label: new_label for new_label, old_label in enumerate(np.unique(labels))}
    return np.array([label_map[label] for label in labels], dtype=int)

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
    csv_path = './Data/set3.csv'
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
    labels_pred = predictor(csv_path)
    rand_index_score = adjusted_rand_score(labels_true, labels_pred)
    print(f'Adjusted Rand Index Score of set3.csv: {rand_index_score:.4f}')


def predictor(csv_path):
    X = load_and_preprocess_predictor(csv_path)
    if X.shape[0] < 5:
        return np.zeros(X.shape[0], dtype=int)

    model = HDBSCAN(
        min_cluster_size=150,
        min_samples=3,
        metric='euclidean',
        cluster_selection_method='eom',
        copy=True,
    )
    labels_pred = model.fit_predict(X)
    non_noise_labels = labels_pred[labels_pred >= 0]

    if len(np.unique(non_noise_labels)) < 2:
        n_clusters = min(X.shape[0], max(2, int(round(X.shape[0] / 600))))
        labels_pred = KMeans(n_clusters=n_clusters, random_state=123, n_init=10).fit_predict(X)

    return renumber_labels(labels_pred)


if __name__=="__main__":
    get_baseline_score()
    evaluate()
