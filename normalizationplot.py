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

normalized_eigenvectors = normalize(eigenvectors)
normalized_laplacian    = normalize(laplacian)

eigenvalues, normalized_laplacian_eigvectors = np.linalg.eig(normalized_laplacian)
normalized_laplacian_eigvectors = normalized_laplacian_eigvectors.astype(np.float64)
both_normalized_eigenvectors = normalize(normalized_laplacian_eigvectors)


def kmeansmethod(eigvectors):
        n = 30
        phis = np.zeros(n)
        best = np.zeros(n)
        for j in range(n):
                kmeans = KMeans(n_clusters=k, max_iter=100, n_init=1).fit(eigvectors)

                cut_edges = 0
                for i in range(edgelist.shape[0]):
                        if (kmeans.labels_[edgelist[0][i]] != kmeans.labels_[edgelist[1][i]]):
                                cut_edges += 1

                counter = collections.Counter(kmeans.labels_)
                smallest = 100000
                for key, value in counter.items():
                        if (value < smallest):
                                smallest = value
                phis[j] = cut_edges / smallest
                best[j] = np.amin(phis[:j+1])
        print(phis)
        return best

phis1 = kmeansmethod(eigenvectors)
phis2 = kmeansmethod(normalized_eigenvectors)
phis3 = kmeansmethod(normalized_laplacian_eigvectors)
phis4 = kmeansmethod(both_normalized_eigenvectors)

plt.plot(phis1, label="not normalized", markerfacecolor='blue')
plt.plot(phis2, label="Normalized eigenvectors", markerfacecolor='red')
plt.plot(phis3, label="Normalized laplacian matrix", markerfacecolor='yellow')
plt.plot(phis4, label="Normalized laplacian matrix and eigenvectors", markerfacecolor='green')
plt.xlabel('Number of iterations')
plt.ylabel('Objective function')
plt.legend()
plt.show()
