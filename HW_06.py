import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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


def main():
    data = pd.read_csv("HW_PCA_SHOPPING_CART_v896.csv")
    columnsCrossCorr(data)
    dataWithoutID = data.iloc[:,1:]
    values = dataWithoutID.values
    kmeans = KMeans(n_clusters=6)
    kmv = kmeans.fit_predict(values)
    
    print("Cluster Centers:", kmeans.cluster_centers_)

    plt.scatter(values[kmv == 0, 0], values[kmv == 0, 1], s=50, c='r', marker='o', edgecolor='black', label='cluster1', alpha=0.25)
    plt.scatter(values[kmv == 1, 0], values[kmv == 1, 1], s=50, c='g', marker='o', edgecolor='black', label='cluster2', alpha=0.25)
    plt.scatter(values[kmv == 2, 0], values[kmv == 2, 1], s=50, c='b', marker='o', edgecolor='black', label='cluster3', alpha=0.25)
    plt.scatter(values[kmv == 3, 0], values[kmv == 3, 1], s=50, c='c', marker='o', edgecolor='black', label='cluster4', alpha=0.25)
    plt.scatter(values[kmv == 4, 0], values[kmv == 4, 1], s=50, c='m', marker='o', edgecolor='black', label='cluster5', alpha=0.25)
    plt.scatter(values[kmv == 5, 0], values[kmv == 5, 1], s=50, c='y', marker='o', edgecolor='black', label='cluster6', alpha=0.25)

    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()
    


if __name__ == '__main__':
    main()