import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import statistics
import timeit

#class that defines a cluster
###########add more comments here
from scipy.cluster import hierarchy


class Cluster:
    def __init__(self, member):
        self.members = [member]
        self.center = []
        for value in range(2, len(member)):
            self.center.append(member[value])

    ### update the center of the cluster after new members have been added
    def updateCenter(self):
        center = []

        # use the mode to find the center instead of the median
        for index in range(2, len(self.members[0])):
            ones = 0
            zeros = 0
            for member in self.members:
                if member[index] == 0:
                    zeros += 1
                else:
                    ones += 1
            if ones > zeros:
                center.append(1)
            else:
                center.append(0)
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
def columnsCrossCorr(dataframe):
    dataWithoutID = dataframe.iloc[:,1:]                    # get the dataframe without the ID column
    matrix = np.ones((len(dataWithoutID.columns), len(dataWithoutID.columns)))
    print(matrix)
    output = open("new_output.csv", "w")
    xIndex = 0
    for firstColumn in dataWithoutID.columns:
        yIndex = 0
        for secondColumn in dataWithoutID.columns:
            if firstColumn == secondColumn:
                output.write('1, ')
                continue
            coef = computeCrossCorrelation(dataWithoutID[firstColumn].tolist(), dataWithoutID[secondColumn].tolist())
            matrix[xIndex][yIndex] = coef
            output.write(str(coef) + ', ')
            yIndex += 1
        xIndex += 1
        output.write("\n")


### Finds the Euchildean distance between two cluster centers
def getDistance(cluster1, cluster2):
    point1 = np.array(cluster1)
    point2 = np.array(cluster2)

    #get sum of the squared value of the two centers
    sum_sq = np.sum(np.square(point1 - point2))

    # Doing squareroot and
    # calculate the Euclidean distance
    distance = (np.sqrt(sum_sq))
    return distance

### Finds the Jaccard similarity and convert to distance
def getDistance2(cluster1, cluster2):
    numerator = 0
    denominator = 0
    for index in range(0, len(cluster1)):
        if cluster1[index] == 1 or cluster2[index] == 1:
            denominator += 1
        if cluster1[index] == 1 and cluster2[index] == 1:
            numerator += 1
    jaccard = numerator / denominator
    distance = 1 - jaccard      # convert jaccard similarity to distance
    return distance


### Agglomerative function takes in the raw data and forms clusters until there
### are only two clusters left (Cluster A and Cluster B)
def agglomerate(data):
    clustersDict = {}

    idx = 0
    for point in data.iterrows():
        clustersDict[idx] = Cluster(point[1])
        idx += 1

    iterTracker=1;
    clusterSizes = []
    while len(clustersDict) > 2:
        bestDistance = float("inf")  # tracks smallest distance btwn clusters
        cluster1Index = 0  # tracks index of one of the clusters with smallest distance
        cluster2Index = 0  # tracks index of other cluster with smallest distance
        firstKey = list(clustersDict.keys())[0]
        secondKey = list(clustersDict.keys())[1]
        for c1Index in range(firstKey, len(clustersDict) + firstKey):
            if c1Index not in clustersDict:
                continue
            for c2Index in range(secondKey, len(clustersDict) + secondKey):
                if c1Index == c2Index or c2Index not in clustersDict:
                    continue
                distance = getDistance(clustersDict[c1Index].center, clustersDict[c2Index].center)
                #print(distance)
                if distance < bestDistance:
                    bestDistance = distance
                    cluster1Index = c1Index
                    cluster2Index = c2Index

        # adds smallest cluster being merged to the small merged cluster list
        if len(clustersDict[cluster1Index].members) <= len(clustersDict[cluster2Index].members):
            clusterSizes.append(len(clustersDict[cluster1Index].members))
        else:
            clusterSizes.append(len(clustersDict[cluster2Index].members))

        # merge the two clusters and delete the second
        clustersDict[cluster1Index].addMembers(clustersDict[cluster2Index].members)
        clustersDict.pop(cluster2Index)

        if len(clustersDict) == 6:
            finalClusters = list(clustersDict.values())
            print("Cluster 1:\n" + str(len(finalClusters[0].members)))
            print("Cluster 2:\n" + str(len(finalClusters[1].members)))
            print("Cluster 3:\n" + str(len(finalClusters[2].members)))
            print("Cluster 4:\n" + str(len(finalClusters[3].members)))
            print("Cluster 5:\n" + str(len(finalClusters[4].members)))
            print("Cluster 6:\n" + str(len(finalClusters[5].members)))

        print("Iteration: " + str(iterTracker))
        iterTracker += 1

    # when we've finished merging, report the biggest clusters that were merged into other clusters
    clusterSizes.sort()  # unsure if I need to sort these before reporting
    print(clusterSizes[-20:])

def main():
    data = pd.read_csv("HW_PCA_SHOPPING_CART_v896.csv")
    #data = pd.read_csv("sample data.csv")
    #data = pd.read_csv("Med_Sample_data.csv")
    # getCrossCorrMatrix(data)
    #columnsCrossCorr(data)
    start = timeit.default_timer()
    #agglomerate(data)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    ###Creates a numPy array and then creates and displays dendrogram
    data_list = data.values.tolist()
    data_array = np.array(data_list)
    fig = ff.create_dendrogram(data_array, color_threshold=205)
    fig.update_layout(width=3000, height=1000)
    fig.show()



if __name__ == '__main__':
    main()