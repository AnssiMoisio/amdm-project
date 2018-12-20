import numpy as np
import scipy.sparse as sparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale, normalize
import pandas as pd
import networkx as nx
import collections
from kmeans import k_means
import matplotlib.pyplot as plt

path      = 'graphs/'
file      = ['ca-GrQc.txt']
k         = 2

edgelist  = pd.read_csv(path + file[0], delimiter=' ', skiprows=1, header=None)
G         = nx.from_pandas_edgelist(edgelist, source=0, target=1, create_using=nx.Graph())
adj_matr  = nx.to_pandas_adjacency(G, dtype=np.float64)
laplacian = sparse.csgraph.laplacian(adj_matr.values, normed=True)

eigenvalues, eigenvectors = np.linalg.eig(laplacian)
eigenvectors = eigenvectors.astype(np.float64)

centroids, clusters = k_means(eigenvectors, k, random_seed=1, num_iters=10, plot=False)

print(centroids)

cut_edges = 0
for i in range(edgelist.shape[0]):
    if (clusters[edgelist[0][i]] != clusters[edgelist[1][i]]):
        cut_edges += 1

counter = collections.Counter(clusters)
smallest = 100000
for key, value in counter.items():
    if (value < smallest):
        smallest = value
phi = cut_edges / smallest
    
print("cuts: ", cut_edges)
print("cluster sizes: ",counter)
print("phi: ", phi)

