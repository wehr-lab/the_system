# Cluster mice depending on their affinity to certain tokensets

import csv
from numpy import genfromtxt
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from matplotlib import pyplot as plt

tok = genfromtxt("/Users/Jonny/Documents/tok_cx.csv",delimiter=',')
names = csv.reader("/Users/Jonny/Documents/mousenames.csv")

# Trim nans
tok = tok[1:,1:]

# Make names
names = list()
with open("/Users/Jonny/Documents/mousenames.csv",'rb') as namecsv:
    nameread = csv.reader(namecsv)
    for row in nameread:
        names.append(row[1])
names = names[1:]

# MDS
# Center tok
tok -= tok.mean()
tok_sim = euclidean_distances(tok)

mds = manifold.MDS(n_components=2, max_iter=3000,dissimilarity="precomputed",n_jobs=4)
pos = mds.fit(tok_sim).embedding_

plt.figure()
plt.scatter(pos[:,0],pos[:,1])
ax = plt.axes()
for i, m in enumerate(names):
    ax.text(pos[i,0],pos[i,1],names[i])