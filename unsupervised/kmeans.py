#!/usr/bin/env python
import sys,os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from mpl_toolkits.mplot3d import Axes3D


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
    authors = None

    try:
        # weird encoding ... 
        headers = [*pd.read_csv(filename, nrows=1)]
        data = pd.read_csv(filename,encoding='iso-8859-1',
        usecols=[c for c in headers if c != 'author'])

        authors = pd.read_csv(filename, encoding='iso-8859-1',usecols=['author'])


    except Exception as e:
        print("Invalid input %s" % e)
        return

    if data.empty:
        print("empty file")
        return

    X = data.to_numpy()
    authors = authors.to_numpy()

    model = KMeans(n_clusters=3,n_init=10)
    model.fit(X)

    centers = np.sort(model.cluster_centers_, axis=0)
    labels = pairwise_distances_argmin(X,centers)

    # just sugar
    markers = ["o", "^", "x"]
    colors  = ["b","g", "r"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print("....")
    for point,label in zip(X, labels):
        x,y,z = point
        marker = markers[label]
        color = colors[label]
        ax.scatter(x,y,z,marker=marker, c=color)

    for point,marker in zip(centers,markers):
        x,y, z = point
        ax.scatter3D(x,y,z,c='k' , s=30, linewidth=0.5)

    ax.set_xlabel('X added')
    ax.set_ylabel('Y deleted')
    ax.set_zlabel('Z commits')

    plt.show()

    # saving file
    def gen(): 
        for name, label, point in zip(authors, labels, X):
            row = list(name) + [label] + list(point)
            yield row

    out = pd.DataFrame(list(gen()), columns=['author','label','added','deleted','commits'])
    outputfile = filename+".output.csv"
    out.to_csv(outputfile, index=False)
    print("take a look to %s" % outputfile)



if __name__ == '__main__':
    if not main():
        sys.exit(1)
