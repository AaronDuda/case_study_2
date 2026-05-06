import functools
import warnings

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import HDBSCAN
from sklearn.metrics.cluster import adjusted_rand_score


def hh_mm_ss2seconds(hh_mm_ss):
    return functools.reduce(lambda acc, x: acc * 60 + x, map(int, hh_mm_ss.split(':')))


def load_data(csv_path):
    return pd.read_csv(csv_path, converters={'SEQUENCE_DTTM': hh_mm_ss2seconds})


def make_features(df, feature_set, weights):
    course_radians = np.deg2rad(df['COURSE_OVER_GROUND'].to_numpy(dtype=float) / 10.0)
    columns = []
    column_weights = []

    if 'time' in feature_set:
        columns.append(df['SEQUENCE_DTTM'].to_numpy(dtype=float))
        column_weights.append(weights['time'])

    if 'position' in feature_set:
        columns.append(df['LAT'].to_numpy(dtype=float))
        columns.append(df['LON'].to_numpy(dtype=float))
        column_weights.append(weights['position'])
        column_weights.append(weights['position'])

    if 'speed' in feature_set:
        columns.append(df['SPEED_OVER_GROUND'].to_numpy(dtype=float))
        column_weights.append(weights['speed'])

    if 'course' in feature_set:
        columns.append(np.sin(course_radians))
        columns.append(np.cos(course_radians))
        column_weights.append(weights['course'])
        column_weights.append(weights['course'])

    X = np.column_stack(columns)
    X = preprocessing.StandardScaler().fit_transform(X)
    return X * np.array(column_weights)


def score_config(datasets, feature_config, min_cluster_size, min_samples, selection_method):
    scores = {}
    clusters = {}
    noise = {}

    for name, df in datasets.items():
        X = make_features(df, feature_config['features'], feature_config['weights'])
        labels = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method=selection_method,
            copy=True,
        ).fit_predict(X)
        scores[name] = adjusted_rand_score(df['VID'].to_numpy(), labels)
        clusters[name] = len(set(labels)) - (1 if -1 in labels else 0)
        noise[name] = float(np.mean(labels == -1))

    return {
        'set1_ari': scores['set1'],
        'set2_ari': scores['set2'],
        'set1_clusters': clusters['set1'],
        'set2_clusters': clusters['set2'],
        'set1_noise': noise['set1'],
        'set2_noise': noise['set2'],
        'features': feature_config['features'],
        'weights': feature_config['weights'],
        'min_cluster_size': min_cluster_size,
        'min_samples': min_samples,
        'cluster_selection_method': selection_method,
    }


def search():
    warnings.filterwarnings('ignore')
    datasets = {
        'set1': load_data('./Data/set1.csv'),
        'set2': load_data('./Data/set2.csv'),
    }
    feature_configs = [
        {'features': ('time', 'position'), 'weights': {'time': 0.25, 'position': 3.0}},
        {'features': ('time', 'position'), 'weights': {'time': 0.5, 'position': 2.0}},
        {'features': ('time', 'position'), 'weights': {'time': 0.5, 'position': 3.0}},
        {'features': ('time', 'position'), 'weights': {'time': 0.5, 'position': 5.0}},
        {'features': ('time', 'position'), 'weights': {'time': 1.0, 'position': 3.0}},
        {'features': ('time', 'position', 'speed'), 'weights': {'time': 0.5, 'position': 3.0, 'speed': 0.25}},
        {'features': ('time', 'position', 'speed'), 'weights': {'time': 1.0, 'position': 3.0, 'speed': 0.5}},
        {'features': ('time', 'position', 'course'), 'weights': {'time': 0.5, 'position': 3.0, 'course': 0.25}},
        {'features': ('time', 'position', 'course'), 'weights': {'time': 1.0, 'position': 3.0, 'course': 0.5}},
        {'features': ('time', 'position', 'speed', 'course'), 'weights': {'time': 1.0, 'position': 1.0, 'speed': 1.0, 'course': 1.0}},
        {'features': ('time', 'position', 'speed', 'course'), 'weights': {'time': 0.5, 'position': 2.0, 'speed': 0.25, 'course': 0.25}},
        {'features': ('time', 'position', 'speed', 'course'), 'weights': {'time': 0.5, 'position': 3.0, 'speed': 0.25, 'course': 0.25}},
        {'features': ('time', 'position', 'speed', 'course'), 'weights': {'time': 1.0, 'position': 3.0, 'speed': 0.5, 'course': 0.5}},
    ]
    min_cluster_sizes = [10, 25, 50, 75, 100, 150, 200]
    min_samples_values = [3, 5, 10]
    selection_methods = ['eom', 'leaf']
    results = []

    for feature_config in feature_configs:
        for min_cluster_size in min_cluster_sizes:
            for min_samples in min_samples_values:
                for selection_method in selection_methods:
                    results.append(score_config(datasets, feature_config, min_cluster_size, min_samples, selection_method))

    strong_set2 = [result for result in results if result['set2_ari'] >= 0.80]
    ranked = sorted(strong_set2 or results, key=lambda result: (result['set1_ari'], result['set2_ari']), reverse=True)
    return ranked


def print_result(result, rank):
    print(f'rank: {rank}')
    print(f'set1_ari: {result["set1_ari"]:.4f}')
    print(f'set2_ari: {result["set2_ari"]:.4f}')
    print(f'set1_clusters: {result["set1_clusters"]}')
    print(f'set2_clusters: {result["set2_clusters"]}')
    print(f'set1_noise: {result["set1_noise"]:.4f}')
    print(f'set2_noise: {result["set2_noise"]:.4f}')
    print(f'features: {result["features"]}')
    print(f'weights: {result["weights"]}')
    print(f'min_cluster_size: {result["min_cluster_size"]}')
    print(f'min_samples: {result["min_samples"]}')
    print(f'cluster_selection_method: {result["cluster_selection_method"]}')
    print()


def main():
    ranked = search()
    for rank, result in enumerate(ranked[:10], start=1):
        print_result(result, rank)


if __name__ == '__main__':
    main()
