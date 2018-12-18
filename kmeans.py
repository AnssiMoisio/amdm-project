import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np

def plotting(data, centroids=None, clusters=None):
    # this function will later on be used for plotting the clusters and centroids.
    # But now we use it to just make a scatter plot of the data
    # Input: the data as an array, cluster means (centroids), cluster assignemnts in {0,1,...,k-1}   
    # Output: a scatter plot of the data in the clusters with cluster means

    # colors
    cmpd = ['orangered','dodgerblue','springgreen']
    cmpcent = ['red','darkblue','limegreen']

    plt.figure(figsize=(5.75,5.25))
    plt.style.use('ggplot')
    plt.title("Data")
    plt.xlabel("feature $x_1$: customers' age")
    plt.ylabel("feature $x_2$: money spent during visit")

    alp = 0.5             # data alpha
    dt_sz = 20            # data point size
    cent_sz = 130         # centroid sz
    
    if centroids is None and clusters is None:
        plt.scatter(data[:,0], data[:,1],s=dt_sz,alpha=alp ,c=cmpd[0])
    if centroids is not None and clusters is None:
        plt.scatter(data[:,0], data[:,1],s=dt_sz,alpha=alp, c=cmpd[0])
        plt.scatter(centroids[:,0], centroids[:,1], marker="x", s=cent_sz, c=cmpcent)
    if centroids is not None and clusters is not None:
        plt.scatter(data[:,0], data[:,1], c=[cmpd[i] for i in clusters], s=dt_sz, alpha=alp)
        plt.scatter(centroids[:,0], centroids[:,1], marker="x", c=cmpcent, s=cent_sz)
    if centroids is None and clusters is not None:
        plt.scatter(data[:,0], data[:,1], c=[cmpd[i] for i in clusters], s=dt_sz, alpha=alp)
    
    plt.show()

def select_centroids(data, k, random_seed=1):   
    # INPUT: N x d data array, k number of clusters. 
    # OUTPUT: k x d array of k randomly assigned mean vectors with d dimensions.
    
    # Random seed will generate exactly same "random" values for each execution.
    # This will ensure similar results between students and avoid confusion.
    np.random.seed(seed=random_seed)
    centroids = np.zeros((k, data.shape[1]))
    for i in range(data.shape[1]):
        centroids[:,i] = np.random.uniform(np.min(data[:,i]), 
                                           np.max(data[:,i]), 
                                           size = (k))
    return centroids

def assign_points(data, centroids):     
    #INPUT: N x d data array, k x d centroids array.
    #OUTPUT: N x 1 array of cluster assignments in {0,...,k-1}.
    clusters = np.zeros(data.shape[0], dtype=np.int32)
    cluster_sizes = np.zeros(centroids.shape[0], dtype=np.int32)
    distance_matrix = np.ndarray((data.shape[0], centroids.shape[0]))
    for i in range(data.shape[0]):
        distances = np.zeros(centroids.shape[0], dtype=np.int32)
        for j in range(centroids.shape[0]):
            distances[j] = np.linalg.norm(data[i] - centroids[j])

        clusters[i] = np.argmin(distances)
        cluster_sizes[clusters[i]] += 1
        distance_matrix[i] = distances

    return clusters

def move_centroids(data, old_centroids, clusters):
    #INPUT:  N x d data array, k x d centroids array, N x 1 array of cluster assignments
    #OUTPUT: k x d array of relocated centroids
    new_centroids = np.zeros(old_centroids.shape)
    for i in range(len(old_centroids)):
        cluster_points = np.array([data[j] for j in range(len(data)) if clusters[j] == i])
        if len(cluster_points) != 0:
            new_centroids[i] = np.mean(cluster_points, axis = 0)
        else:
            new_centroids[i] = old_centroids[i]
    return new_centroids

def k_means(data, k, random_seed=1, num_iters=10, plot=True):
    #INPUT: N x d data array, k number of clusters, number of iterations, boolean plot.
    #OUTPUT: N x 1 array of cluster assignments.
    centroids = select_centroids(data, k, random_seed)
    for i in range(11):
        clusters = assign_points(data, centroids)
        if plot==True and i < 3:
            plotting(data,centroids,clusters)
        centroids = move_centroids(data, centroids, clusters)
    return centroids, clusters
'''
def kmeans(data):
    clusters = assign_points(data, centroids)
    plotting(data, centroids, clusters)

    new_centroids = move_centroids(data, centroids, clusters)
    plotting(data, new_centroids, clusters)

    centroids,clusters = k_means(data, 2)
    print("The final cluster mean values are:", centroids)
'''