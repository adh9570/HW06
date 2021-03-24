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
def getCrossCorrMatrix(dataframe):
    dataWithoutID = dataframe.iloc[:,1:]                    # get the dataframe without the ID column
    data = dataWithoutID.values                             # turn dataframe into 2D array for use in cross correlation computations

    matrix = np.empty((len(data), len(data)))               # create an empty matrix that will hold our cross correlation values
    for rowIndexX in range(0, len(data)):                   # go through each row of the data (called X row)
        dataRowX = data[rowIndexX]
        for rowIndexY in range(0, len(data)):               # compare to every other row of the data (called Y row)
            if rowIndexX == rowIndexY:                      # skip comparing a row to itself
                continue                                    # matrix is automatically populated with zeros, which is the cross coeff of an attribute and itself
            dataRowY = data[rowIndexY]
            matrix[rowIndexX][rowIndexY] = computeCrossCorrelation(dataRowX, dataRowY)  # add cross correlation to the matrix at position [X, Y]
    print(matrix)
    return matrix


def main():
    # data = pd.read_csv("HW_PCA_SHOPPING_CART_v896.csv")
    data = pd.read_csv("sample data.csv")
    getCrossCorrMatrix(data)
    

if __name__ == '__main__':
    main()