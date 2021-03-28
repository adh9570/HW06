import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

def kmeansCluster(values):
    kmeans = KMeans(n_clusters=6)
    kmv = kmeans.fit_predict(values)
    
    print("Cluster Centers:", kmeans.cluster_centers_)
    centers = kmeans.cluster_centers_
    centersFile = open("centers.csv", "w")
    for i in centers:
        for j in i:
            centersFile.write(str(j) + ', ')
        centersFile.write('\n')

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

    plt.scatter(values[kmv == 0, 0], values[kmv == 0, 1], s=50, c='r', marker='o', label='cluster1', alpha=0.25)
    plt.scatter(values[kmv == 1, 0], values[kmv == 1, 1], s=50, c='g', marker='o', label='cluster2', alpha=0.25)
    plt.scatter(values[kmv == 2, 0], values[kmv == 2, 1], s=50, c='b', marker='o', label='cluster3', alpha=0.25)
    plt.scatter(values[kmv == 3, 0], values[kmv == 3, 1], s=50, c='c', marker='o', label='cluster4', alpha=0.25)
    plt.scatter(values[kmv == 4, 0], values[kmv == 4, 1], s=50, c='m', marker='o', label='cluster5', alpha=0.25)
    plt.scatter(values[kmv == 5, 0], values[kmv == 5, 1], s=50, c='y', marker='o', label='cluster6', alpha=0.25)

    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()



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


def main():
    data = pd.read_csv("HW_PCA_SHOPPING_CART_v896.csv")
    dataWithoutID = data.iloc[:,1:]
    columnsCrossCorr(dataWithoutID)
    values = dataWithoutID.values
    kmeansCluster(values)


if __name__ == '__main__':
    main()