import pandas as pd
import numpy as np
import statistics


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


class Cluster:
    def __init__(self, member):
        self.members = [member]
        self.center = []
        for value in range(2, len(member)):
            self.center.append(member[value])

    ### update the center of the cluster after new members have been added
    def updateCenter(self):
        center = []

        # median calculation, removed bc of later updates/clarifications to hw
        # for index in range(2, len(self.members[0])):
        #     total = 0
        #     for member in self.members:
        #         total+=member[index]
        #     total = total/len(self.members)     # find median of the index
        #     center.append(total)
        # self.center = center

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


### find the L1 norm between two clusters
def L1Norm(cluster1, cluster2):
    distance = 0
    for index in range(0, len(cluster1)):
        distance += abs(cluster1[index] - cluster2[index])
    return distance


### Finds the Jaccard similarity and convert to distance
def getDistance(cluster1, cluster2):
    point1 = np.array(cluster1)
    point2 = np.array(cluster2)

    sum_sq = np.sum(np.square(point1 - point2))

    # Doing squareroot and
    # printing Euclidean distance
    distance = (np.sqrt(sum_sq))
    return distance


### Agglomerative function takes in the raw data and forms clusters until there
### are only two clusters left (Cluster A and Cluster B)
def agglomerate(data):
    clustersDict = {}

    idx = 0
    for point in data.iterrows():
        clustersDict[idx] = Cluster(point[1])
        idx += 1

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

        if len(clustersDict) == 2:
            finalClusters = list(clustersDict.values())
            print("Cluster A:\n" + str(finalClusters[0].members))
            print("\nCluster B:\n" + str(finalClusters[1].members))

    # when we've finished merging, report the biggest clusters that were merged into other clusters
    clusterSizes.sort()  # unsure if I need to sort these before reporting
    print(clusterSizes[-20:])






def main():
    data = pd.read_csv("HW_PCA_SHOPPING_CART_v896.csv")
    # data = pd.read_csv("sample data.csv")
    # getCrossCorrMatrix(data)
    columnsCrossCorr(data)
    agglomerate(data)


if __name__ == '__main__':
    main()