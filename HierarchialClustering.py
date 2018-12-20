import numpy as np

class HierarchialClustering:

    def __init__(self, data, k = 2):
        self.k        = k
        self.data     = data
        self.clusters = [np.array(row) for row in data]

    def merge(self, idx1, idx2):
        new_set = np.vstack((self.clusters[idx1], self.clusters[idx2]))
        self.clusters.pop(max(idx1, idx2))
        self.clusters.pop(min(idx1, idx2))
        self.clusters.append(new_set)

    def compute(self):
        set1, set2 = self.clusters[0], self.clusters[1]
        minimum_distance = self.minimum_distance(set1, set2)
        while len(self.clusters) > self.k:
            for i in len(self.clusters):
                for j in len(self.clusters):
                    distance = self.minimum_distance(self.clusters[i], self.clusters[j])
                    if i is not j and distance < minimum_distance:
                        set1, set2 = self.clusters(i), self.clusters(j)
                        minimum_distance = distance
            self.merge(i, j)
        return self.clusters

    def minimum_distance(self, set1, set2):
        print(set1.T.shape)
        avg1 = np.average(set1.T)
        avg2 = np.average(set2.T)
        print(avg1.shape)
        print(avg2.shape)
        return np.linalg.norm(avg1, avg2)
