import numpy as np
import scipy.sparse as sparse
from sklearn.cluster import KMeans
import pandas as pd
import networkx as nx

path      = 'graphs/'
file      = ['ca-GrQc.txt']
k         = 4

edgelist  = pd.read_csv(path + file[0], delimiter=' ', skiprows=1, header=None)
G         = nx.from_pandas_edgelist(edgelist, source=0, target=1, create_using=nx.Graph())
adj_matr  = nx.to_pandas_adjacency(G, dtype=np.float64)
laplacian = sparse.csgraph.laplacian(adj_matr.values, normed=True)

eigenvalues, eigenvectors = np.linalg.eig(laplacian)
kmeans = KMeans(n_clusters=k, max_iter=3000).fit(eigenvectors.astype(np.float64))

print(kmeans.labels_)