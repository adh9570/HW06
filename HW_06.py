import timeit
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy


class Cluster:
    def __init__(self, member):
        self.members = [member]
        self.center = []
        for value in range(1, len(member)):
            self.center.append(member[value])

    ### update the center of the cluster after new members have been added
    def updateCenter(self):
        center = []
        sum_list=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # use the mode to find the center instead of the median
        for index in range(0, len(self.members[0])-1):
            for member in self.members:
                sum_list[index]+=member[index]
        for value in sum_list:
            center.append(value/len(self.members))
        self.center = center

    ### adds the members from one cluster to the current cluster
    def addMembers(self, newMembers):
        count = 0
        for member in range(0, len(newMembers)):
            self.members.append(newMembers[member])
            count += 1
        self.updateCenter()


### Derived from slide 44 of Distance Metrics part A slides
### Takes in two arrays, attributes S and T, and uses the average of all of
### the values in each attribute, along with the standard deviation of the 
### values to calculate the cross correlation coefficient between the two 
### attributes
def computeCrossCorrelation(attributeS, attributeT):
    N = len(attributeS)

    avgS = np.sum(attributeS) / len(attributeS)
    avgT = np.sum(attributeT) / len(attributeT)

    stdDevS = statistics.stdev(attributeS)
    stdDevT = statistics.stdev(attributeT)

    crossCorrSum = 0
    for k in range(0, N):
        crossCorrSum += ((attributeS[k] - avgS)/stdDevS) * ((attributeT[k] - avgT)/stdDevT)

    return crossCorrSum/N

### Go through every row in the data and compare it to every other row of data,
### compute the cross correlation, and put that value into an NxN matrix of all
### of the cross correlation values
def columnsCrossCorr(data):
    matrix = np.ones((len(data.columns), len(data.columns)))
    output = open("new_output.csv", "w")
    xIndex = 0
    for firstColumn in data.columns:
        yIndex = 0
        for secondColumn in data.columns:
            if firstColumn == secondColumn:
                output.write('1, ')
                continue
            coef = computeCrossCorrelation(data[firstColumn].tolist(), data[secondColumn].tolist())
            matrix[xIndex][yIndex] = coef
            output.write(str(coef) + ', ')
            yIndex += 1
        xIndex += 1
        output.write("\n")


### Takes in the data without the ID column and runs it through the sklearn
### algorithm to cluster the data into six clusters. Once the clusters have 
### been formed, the centers of the 
def kmeansCluster(values):
    kmeans = KMeans(n_clusters=6)
    kmv = kmeans.fit_predict(values)
    
    print("Cluster Centers:", kmeans.cluster_centers_)
    centers = kmeans.cluster_centers_

    # write the cluster centers to a file to be able to more clearly read the data
    centersFile = open("centers.csv", "w")
    for i in centers:
        for j in i:
            centersFile.write(str(j) + ', ')
        centersFile.write('\n')

    # count the number of datapoints in each cluster then prints out the cluster sizes
    clusterSizes = [0] * 6
    for value in kmv:
        if value == 0:
            clusterSizes[0] += 1
        elif value == 1:
            clusterSizes[1] += 1
        elif value == 2:
            clusterSizes[2] += 1
        elif value == 3:
            clusterSizes[3] += 1
        elif value == 4:
            clusterSizes[4] += 1
        elif value == 5:
            clusterSizes[5] += 1
        elif value == 6:
            clusterSizes[6] += 1
        
    print("Cluster Sizes", clusterSizes)

    # plots the six different clusters, each in a different color, in a scatter plot
    # the alpha value is purposely low to be able to visualize areas of the graph 
    # where there is a higher density of datapoints
    plt.scatter(values[kmv == 0, 0], values[kmv == 0, 1], s=50, c='r', marker='o', label='Cluster A', alpha=0.25)
    plt.scatter(values[kmv == 1, 0], values[kmv == 1, 1], s=50, c='g', marker='o', label='Cluster B', alpha=0.25)
    plt.scatter(values[kmv == 2, 0], values[kmv == 2, 1], s=50, c='b', marker='o', label='Cluster C', alpha=0.25)
    plt.scatter(values[kmv == 3, 0], values[kmv == 3, 1], s=50, c='c', marker='o', label='Cluster D', alpha=0.25)
    plt.scatter(values[kmv == 4, 0], values[kmv == 4, 1], s=50, c='m', marker='o', label='Cluster E', alpha=0.25)
    plt.scatter(values[kmv == 5, 0], values[kmv == 5, 1], s=50, c='y', marker='o', label='Cluster F', alpha=0.25)

    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()


### Find the Euchildean distance between two cluster centers
def getDistance(cluster1, cluster2):
    point1 = np.array(cluster1)
    point2 = np.array(cluster2)

    sum_sq = np.sum(np.square(point1 - point2))     # get sum of the squared value of the two centers

    distance = (np.sqrt(sum_sq))                    # Doing squareroot and calculate the Euclidean distance
    return distance


### Agglomerative function takes in the raw data and forms clusters until there
### are only two clusters left (Cluster A and Cluster B)
def agglomerate(data):
    clustersDict = {}

    idx = 0
    for point in data.iterrows():
        clustersDict[idx] = Cluster(point[1])
        idx += 1

    iterTracker=1
    clusterSizes = []
    while len(clustersDict) > 2:
        bestDistance = float("inf")                 # tracks smallest distance btwn clusters
        cluster1Index = 0                           # tracks index of one of the clusters with smallest distance
        cluster2Index = 0                           # tracks index of other cluster with smallest distance
        firstKey = list(clustersDict.keys())[0]
        secondKey = list(clustersDict.keys())[1]
        bound = list(clustersDict.keys())

        for c1Index in range(firstKey, bound[-1]):  #len(clustersDict) + firstKey):
            if c1Index not in clustersDict:
                continue
            for c2Index in range(secondKey, bound[-1]): #len(clustersDict) + secondKey):
                if c1Index >= c2Index or c2Index not in clustersDict:
                    continue
                distance = getDistance(clustersDict[c1Index].center, clustersDict[c2Index].center)
                if distance <= bestDistance:
                    bestDistance = distance
                    cluster1Index = c1Index
                    cluster2Index = c2Index

        # adds smallest cluster being merged to the small merged cluster list
        if len(clustersDict[cluster1Index].members) < len(clustersDict[cluster2Index].members):
            clusterSizes.append(len(clustersDict[cluster1Index].members))
            clustersDict[cluster2Index].addMembers(clustersDict[cluster1Index].members)
            clustersDict.pop(cluster1Index)
        else:
            clusterSizes.append(len(clustersDict[cluster2Index].members))
            clustersDict[cluster1Index].addMembers(clustersDict[cluster2Index].members)
            clustersDict.pop(cluster2Index)

        if len(clustersDict) == 6:
            finalClusters = list(clustersDict.values())
            print("Cluster 1:\n   Size:" + str(len(finalClusters[0].members)))
            print("   Center: "+str(finalClusters[0].center))
            print("Cluster 2:\n   Size:" + str(len(finalClusters[1].members)))
            print("   Center: " + str(finalClusters[1].center))
            print("Cluster 3:\n   Size:" + str(len(finalClusters[2].members)))
            print("   Center: " + str(finalClusters[2].center))
            print("Cluster 4:\n   Size:" + str(len(finalClusters[3].members)))
            print("   Center: " + str(finalClusters[3].center))
            print("Cluster 5:\n   Size:" + str(len(finalClusters[4].members)))
            print("   Center: " + str(finalClusters[4].center))
            print("Cluster 6:\n   Size:" + str(len(finalClusters[5].members)))
            print("   Center: " + str(finalClusters[5].center))

        print("Iteration: " + str(iterTracker))
        iterTracker += 1

    # when we've finished merging, report the clusters that were merged into other clusters
    print(clusterSizes[-18:])


def main():
    data = pd.read_csv("HW_PCA_SHOPPING_CART_v896.csv")

    ### Compute cross-correlation matrix
    dataWithoutID = data.iloc[:,1:]
    columnsCrossCorr(dataWithoutID)

    ### Compute KMeans clustering
    values = dataWithoutID.values
    kmeansCluster(values)

    ### Agglomerate
    # data = pd.read_csv("sample data.csv")
    #data = pd.read_csv("Med_Sample_data.csv")
    start = timeit.default_timer()
    agglomerate(data)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    ###Creates a numPy array and then creates and displays dendrogram
    data_list = data.values.tolist()
    data_array = np.array(data_list)
    fig = ff.create_dendrogram(data_array, color_threshold=220)
    fig.update_layout(width=1000, height=1000)
    fig.show()

if __name__ == '__main__':
    main()