import numpy as np
import scipy.sparse as sparse
import pandas as pd
import networkx as nx

file      = ['ca-AstroPh.txt']
edgelist  = pd.read_csv(file[0], delimiter='\t', skiprows=4, header=None)
G         = nx.from_pandas_edgelist(edgelist, source=0, target=1, create_using=nx.DiGraph())
adj_matr  = nx.to_pandas_adjacency(G, dtype=np.float32)

"""
N = len(edgelist[0].unique())
in_degrees  = {}
out_degrees = {}
for i in edgelist[0].unique():
    in_degrees[i]  = 0
    out_degrees[i] = 0

for edge in edgelist.values:
    in_degrees[edge[1]]  += 1
    out_degrees[edge[0]] += 1
"""

# seuraavaksi pitäisi kertoa kaikki arvot -1/np.sqrt(out_degree[vi] * out_degree[vj])
laplacian = sparse.csgraph.laplacian(adj_matr.values, normed=True)

print(laplacian)