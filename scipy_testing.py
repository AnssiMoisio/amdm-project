import numpy as np
import scipy.sparse as sparse
import pandas as pd

file             = ['ca-AstroPh.txt']
edgelist         = pd.read_csv(file[0], delimiter='\t', skiprows=4, header=None)
adjacency_matrix = np.empty()

print(adjacency_matrix)