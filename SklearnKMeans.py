#Team Members:
# Dhruv Bajpai - dbajpai - 6258833142
# Anupam Mishra - anupammi - 2053229568

import pandas as pd
import sys
import numpy as np
from sklearn.cluster import KMeans

def main():
    test_data = sys.argv[1]
    data = pd.read_csv(test_data,sep=',', header=None)
    data.columns = ["x","y"]
    points  = pd.DataFrame(data)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(points)
    centroids = kmeans.cluster_centers_
    print(centroids)

if __name__ == '__main__':
    main()
