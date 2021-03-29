import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.cluster import KMeans

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
        # use the mean to find the center
        for index in range(0, len(self.members[0])-1):
            for member in self.members:                         # go through each member in the cluster
                sum_list[index]+=member[index+1]                # add each member's attributes to a total sum
        for value in sum_list:                                  # divide each value in the sum tracker by the number
            center.append(value/len(self.members))              # of members whose attributes contributed to the sum
        self.center = center

    ### adds the members from one cluster to the current cluster
    def addMembers(self, newMembers):
        count = 0
        for member in range(0, len(newMembers)):
            self.members.append(newMembers[member])
            count += 1
        self.updateCenter()                                     # update the center of the cluster to account for new members


### Derived from slide 44 of Distance Metrics part A slides
### Takes in two arrays, attributes S and T, and uses the average of all of
### the values in each attribute, along with the standard deviation of the 
### values to calculate the cross correlation coefficient between the two 
### attributes
def computeCrossCorrelation(attributeS, attributeT):
    avgS = np.sum(attributeS) / len(attributeS)                 # get the average of all of the values in attribute S
    avgT = np.sum(attributeT) / len(attributeT)                 # get the average of all of the values in attribute T

    stdDevS = statistics.stdev(attributeS)                      # get the standard deviation of the values in attribute S
    stdDevT = statistics.stdev(attributeT)                      # get the standard deviation of the values in attribute T

    N = len(attributeS)
    
    crossCorrSum = 0
    for k in range(0, N):                                       # equation on slide 44 of Distance Metrics part A
        crossCorrSum += ((attributeS[k] - avgS)/stdDevS) * ((attributeT[k] - avgT)/stdDevT)
    return crossCorrSum/N


### Go through every row in the data and compare it to every other row of data,
### compute the cross correlation, and put that value into an NxN matrix of all
### of the cross correlation values
def columnsCrossCorr(data):
    matrix = np.ones((len(data.columns), len(data.columns)))    # create tracker matrix to hold info on cross correlation coefficient calculations between attributes
    output = open("Cross_Correlation_Coefficients.csv", "w")    # create a file to hold the cross correlation coefficients between attributes
    xIndex = 0                                                  # index for tracking the first attribute in the matrix
    for firstColumn in data.columns:                            # go through each column and calculate the cross corr coef with every other attribute
        yIndex = 0                                              # index for tracking the second attribute in the matrix
        for secondColumn in data.columns:
            if firstColumn == secondColumn:                     # if calculating the cross corr coef of one attribute with itself, the value will always be 1
                output.write('1, ')                             # write a 1 to the output file (do not need to add a 1 to the matrix because the matrix is pre-filled with 1s
                continue
            coef = computeCrossCorrelation(data[firstColumn].tolist(), data[secondColumn].tolist())     # get cross corr coefficient of two attributes
            matrix[xIndex][yIndex] = coef                       # add the cross corr coef to the tracker matrix in the position [first attribute][second attribute]
            output.write(str(coef) + ', ')                      # add the cross corr coef to the tracking csv along with a comma deliminator
            yIndex += 1
        xIndex += 1
        output.write("\n")                                      # add a new line after all of the calculations are finished for the first attribute 


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

    # get sum of the squared value of the two centers
    sum_sq = np.sum(np.square(point1 - point2))

    # Doing squareroot and calculate the Euclidean distance
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

    iterTracker=1
    clusterSizes = []
    while len(clustersDict) > 2:
        bestDistance = float("inf")                 # tracks smallest distance btwn clusters
        cluster1Index = 0                           # tracks index of one of the clusters with smallest distance
        cluster2Index = 0                           # tracks index of other cluster with smallest distance
        firstKey = list(clustersDict.keys())[0]     # lowest key in the clusters dictionary
        secondKey = list(clustersDict.keys())[1]    # second lowest key in the clusters dictionary
        keys = list(clustersDict.keys())
        for c1Index in range(firstKey, keys[-1]):   # go through each member in the cluster and compare its center to every other cluster center
            if c1Index not in clustersDict:
                continue
            for c2Index in range(secondKey, keys[-1]):
                if c1Index >= c2Index or c2Index not in clustersDict:
                    continue
                distance = getDistance(clustersDict[c1Index].center, clustersDict[c2Index].center)  # get the distance between clusters 1 and 2
                if distance <= bestDistance:        # if this distance is equal to or better than our best distance
                    bestDistance = distance         # update to track the new closest clusters and the distance
                    cluster1Index = c1Index         # between them
                    cluster2Index = c2Index

        # adds smallest cluster being merged to the small merged cluster list and merge into the larger cluster
        if len(clustersDict[cluster1Index].members) < len(clustersDict[cluster2Index].members):
            clusterSizes.append(len(clustersDict[cluster1Index].members))
            clustersDict[cluster2Index].addMembers(clustersDict[cluster1Index].members)
            clustersDict.pop(cluster1Index)
        else:
            clusterSizes.append(len(clustersDict[cluster2Index].members))
            clustersDict[cluster1Index].addMembers(clustersDict[cluster2Index].members)
            clustersDict.pop(cluster2Index)

        # Prints prototype of each of the final 6 clusters
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

        #print("Iteration: " + str(iterTracker))
        #iterTracker += 1

    # when we've finished merging, report the clusters that were merged into other clusters
    print(clusterSizes[-18:])

    
def main():
    data = pd.read_csv("HW_PCA_SHOPPING_CART_v896.csv") # read the shopping data into a dataframe

    ### Compute cross-correlation matrix
    dataWithoutID = data.iloc[:,1:]                     # get the data from the dataframe without the ID attached
    columnsCrossCorr(dataWithoutID)                     # compute the cross correlation coefficients for every attribute with every other attribute

    ### Compute KMeans clustering
    values = dataWithoutID.values                       # get the list of values from the data (in array format rather than dataframe)
    kmeansCluster(values)                               # cluster the values in the data using KMeans

    ### Agglomerate
    agglomerate(data)                                   # cluster the data using agglomeration

    ###Creates a numPy array and then creates and displays dendrogram
    data_list = data.values.tolist()
    data_array = np.array(data_list)                    # get an array of the values in the data
    fig = ff.create_dendrogram(data_array, color_threshold=220) # send the data values array in to create a dendogram
    fig.update_layout(width=1000, height=1000)
    fig.show()                                          # display the dendogram


if __name__ == '__main__':
    main()
