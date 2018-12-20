import numpy as np
import scipy.sparse as sparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale, normalize
import pandas as pd
import networkx as nx
import collections
from kmeans import k_means
import matplotlib.pyplot as plt
import math


def calculatePhi(graph, k):
    path      = 'graphs/'
    file      = [graph]

    edgelist  = pd.read_csv(path + file[0], delimiter=' ', skiprows=1, header=None)
    G         = nx.from_pandas_edgelist(edgelist, source=0, target=1, create_using=nx.Graph())
    adj_matr  = nx.to_pandas_adjacency(G, dtype=np.float64)
    laplacian = sparse.csgraph.laplacian(adj_matr.values, normed=True)
    laplacian   = normalize(laplacian)

    eigenvalues, eigenvectors = np.linalg.eig(laplacian)
    kmeans = KMeans(n_clusters=k, max_iter=3000).fit(eigenvectors.astype(np.float64))

    cut_edges = 0
    for i in range(edgelist.shape[0]):
            if (kmeans.labels_[edgelist[0][i]] != kmeans.labels_[edgelist[1][i]]):
                    cut_edges += 1

    counter = collections.Counter(kmeans.labels_)
    smallest = math.inf
    for key, value in counter.items():
            if (value < smallest):
                    smallest = value
    phi = cut_edges / smallest

    return phi

phi1 = calculatePhi('ca-GrQc.txt', 2)
print(phi1)
phi2 = calculatePhi('Oregon-1.txt', 5)
print(phi2)
phi3 = calculatePhi('soc-Epinions1.txt', 10)
print(phi3)
phi4 = calculatePhi('web-NotreDame.txt', 20)
print(phi4)
phi5 = calculatePhi('roadNet-CA', 50)
print(phi5)



plt.plot(phi1, marker='.',color='red', label='ca-GrQc')
plt.plot(phi2, marker='.',color='yellow', label='Oregon-1')
plt.plot(phi3, marker='.',color='blue', label='soc-Epinions1')
plt.plot(phi4, marker='.',color='green', label='web-NotreDame')
plt.plot(phi5, marker='.',color='magenta', label='roadNet-CA')
plt.ylabel('Objective function')
plt.legend()

plt.show()
