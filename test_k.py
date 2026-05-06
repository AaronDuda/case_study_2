import pandas as pd, numpy as np, functools
from sklearn import preprocessing

def hh_mm_ss2seconds(hh_mm_ss):
    return functools.reduce(lambda acc, x: acc*60 + x, map(int, hh_mm_ss.split(':')))

for name in ['set1.csv', 'set2.csv']:
    df = pd.read_csv(f'./Data/{name}', converters={'SEQUENCE_DTTM': hh_mm_ss2seconds})
    print(f'\n{name} (true K={df["VID"].nunique()}):')
    print(f'  LAT range: {df["LAT"].min():.4f} - {df["LAT"].max():.4f}  spread={df["LAT"].max()-df["LAT"].min():.4f}')
    print(f'  LON range: {df["LON"].min():.4f} - {df["LON"].max():.4f}  spread={df["LON"].max()-df["LON"].min():.4f}')
    print(f'  SPEED range: {df["SPEED_OVER_GROUND"].min():.2f} - {df["SPEED_OVER_GROUND"].max():.2f}')
    print(f'  COURSE range: {df["COURSE_OVER_GROUND"].min():.2f} - {df["COURSE_OVER_GROUND"].max():.2f}')
    
    # per-VID centroids
    centroids = df.groupby('VID')[['LAT','LON']].mean()
    print(f'\n  Per-VID centroids:')
    print(centroids.to_string())
    
    # pairwise centroid distances
    from scipy.spatial.distance import pdist
    dists = pdist(centroids.values)
    print(f'\n  Centroid pairwise distances: min={dists.min():.4f} mean={dists.mean():.4f} max={dists.max():.4f}')
    
    # within-VID spread
    spreads = df.groupby('VID')[['LAT','LON']].std().mean(axis=1)
    print(f'  Within-VID spread (std): min={spreads.min():.4f} mean={spreads.mean():.4f} max={spreads.max():.4f}')
