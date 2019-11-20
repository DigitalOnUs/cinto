#!/usr/bin/env python
import sys,os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import MeanShift, estimate_bandwidth


# sample header 
# author     string
# added      integer
# deleted    integer
# commits    integer

def main():
    # args
    if len(sys.argv) < 2:
        print("Usage: %s <user_file>" % sys.argv[0])
        return
    
    filename = sys.argv[1]
    data = None

    try:
        # weird encoding ... 
        headers = [*pd.read_csv(filename, nrows=1)]
        data = pd.read_csv(filename,encoding='iso-8859-1',
         usecols=[c for c in headers if c != 'author'])

    except Exception as e:
        print("Invalid input %s" % e)
        return

    if data.empty:
        print("empty file")
        return

    raw = data.to_numpy()
    # The bandwidth is the distance/size scale of the kernel function, 
    # i.e. what the size of the “window” is across which you calculate the mean.
    for b in raw:
        print("....", b)

    bandwidth = estimate_bandwidth(raw, quantile=0.2, n_samples=500, n_jobs=2)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(raw)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)
    print(cluster_centers)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(labels_unique)

    for plot in raw:
        x,y,z = plot
        ax.scatter(x,y,z,marker='^')

    # plotting centers
    for c in cluster_centers:
        x,y, z = c
        ax.scatter(x,y,z, marker='o')

    ax.set_xlabel('X added')
    ax.set_ylabel('Y deleted')
    ax.set_zlabel('Z commits')

    plt.show()


if __name__ == '__main__':
    if not main():
        sys.exit(1)
