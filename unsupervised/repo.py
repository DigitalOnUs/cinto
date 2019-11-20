#!/usr/bin/env python3

import argparse
import sys
import numpy as np
import pandas as pd
import random

from miscs import get_data, split
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn import preprocessing

import matplotlib.pyplot as plt
import matplotlib._color_data as mcd

def main():
    # args
    parser = argparse.ArgumentParser()
    # parsing different files
    parser.add_argument("-c", "--coupling", help="extra file for coupling", type=str)
    parser.add_argument("-i", "--ignore", help="except columns from csv", action='append')
    parser.add_argument("-k", "--key", help="merge key for coupling if not provided using the first column")
    parser.add_argument("-s", "--suffix", help="output suffix for input files input.output.csv",
                        type=str, default=".output.csv")
    # positional
    parser.add_argument("--plot", help="plot the calculus", action="store_true")
    parser.add_argument("-a","--algo", help="algorithm to perform the clustering default kmeans", default="kmeans")
    parser.add_argument("files", nargs="*")

    args = parser.parse_args()

    if not len(args.files):
        print("Please provide a list of csv\n %s input.csv" % sys.argv[0])
        return

    values, meta = None, None
    headers = []
    filename = args.files[0]
    # reading inputs
    try:
        src = get_data(filename)
        if args.coupling:
            coupling = get_data(args.coupling)
            #dummy req first column is the index
            key = args.key
            if not key:
                if src.columns[0] != coupling.columns[0]:
                    print("please provide the key to couple")
                    return
                key = src.columns[0]
            src = src.join(coupling.set_index(key), on=key)

        values, meta, headers = split(src, args.ignore)
    except Exception as e:
        print("Phase 1 error %s" % e)
        return

    if values.empty:
        print("empty file %s " % filename)
        return

    X = values.to_numpy()
    X = np.nan_to_num(X)
    M = None
    if not meta.empty:
        M = meta.to_numpy()

    model = KMeans(n_clusters=3, n_init=10)
    model.fit(X)

    # output with centroid
    centers = np.sort(model.cluster_centers_, axis=0)
    labels = pairwise_distances_argmin(X,centers)

    # creating output file with the new classes
    # saving file
    def gen():
        if M.any():
            for m, var, label in zip(M, X, labels):
                row = list(m) + list(var) + [label]
                yield row
        else:
            for var, label in zip(X, labels):
                row = list(var) + [label]
                yield row

    columns = headers + ["robot-type"]
    if args.ignore:
        columns = args.ignore + columns

    out = pd.DataFrame(list(gen()), columns=columns)
    outputfile = filename + args.suffix
    out.to_csv(outputfile, index=False)
    print("take a look to %s" % outputfile)

    # plotting 2D
    if args.plot:
        from sklearn.decomposition import IncrementalPCA
        ipca = IncrementalPCA(n_components=2, batch_size=10)
        X_scale = preprocessing.scale(X)
        X_pca = ipca.fit_transform(X_scale)
        display_name = [ 'cluster%d' % i for i in range(len(centers)) ]
        colors = [name for name in mcd.CSS4_COLORS]
        random.shuffle(colors)
        plt.figure()
        for point, label in zip(X_pca,labels):
            if label > len(colors):
                print("cannot plot this huge cluster ... skipping label %", display_name[label])
                continue
            x, y = point
            color = colors[label]
            plt.scatter(x,y, color=color, lw=2, label=display_name[label])

        # clusters

        plt.show()

if __name__ == '__main__':
    if not main():
        sys.exit(1)