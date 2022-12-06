import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def ncluster(n,df):
    df = StandardScaler().fit_transform(df)
    kmeans = KMeans(n_clusters=n).fit(df)
    centroids = kmeans.cluster_centers_
    print(centroids)
    return kmeans

    # plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
    # plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    # plt.show()

def optimalclusters(df):
    sse = []
    df = StandardScaler().fit_transform(df)
    kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "random_state": 1,
    }
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
    print(sse)
    return np.argmin(sse)

if __name__ == "__main__":
    drug_data = pd.read_pickle("drug_related.pkl")
    selecteddata = drug_data[['dispatch_response_seconds_qy', 'incident_response_seconds_qy', 'incident_travel_tm_seconds_qy','zipcode']].copy()
    selecteddata.dropna(axis=0,inplace=True)
    zipcodelabels = selecteddata['zipcode']
    selecteddata.drop('zipcode',axis = 1, inplace=True)
    # print(optimalclusters(selecteddata))
    optimal = 5
    kmeans = ncluster(df = selecteddata, n = optimal)
    print(kmeans.labels_)
