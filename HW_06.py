import pandas as pd
import numpy as np
import statistics

def getAverage(set):
    total = 0
    for value in set:
        total += value
    return total/len(set)


### Derived from slide 44 of Distance Metrics part A slides
### TODO: currently this is a generic implementation of the algorithm, will 
### need to be fleshed out more when we figure out exactly what S and T are
def computeCrossCorrelation(s, t):
    # p(s, t) = 1/N * sum(from 1->N)[((sk - us)/os) * ((tk - ut)/ot)]
    # where N is ??? the number of elements in the set ??
    # where us is the average of S and ut is the average of T
    # where os is the standard deviation of S and same for ot and T
    N = len(s)                  # TODO: THIS IS A GUESS FIX IT ALL
    # avgS = getAverage(s)
    # avgT = getAverage(t)
    avgS = np.sum(s) / len(s)
    avgT = np.sum(t) / len(t)

    stdDevS = statistics.stdev(s)
    stdDevT = statistics.stdev(t)

    crossCorrSum = 0
    for k in range(0, N):
        crossCorrSum += ((s[k] - avgS)/stdDevS) * ((t[k] - avgT)/stdDevT)

    return crossCorrSum/N


### Go through every row in the data and compare it to every other row of data,
### compute the cross correlation, and put that value into an NxN matrix of all
### of the cross correlation values
def getCrossCorrMatrix(data):
    matrix = np.empty((len(data), len(data)))               # create an empty matrix that will hold our cross correlation values
    for rowIndexX in range(0, len(data)):                   # go through each row of the data (called X row)
        dataRowX = data.iloc[rowIndexX][1:]                 # get the X row exclusing the ID
        for rowIndexY in range(0, len(data)):               # compare to every other row of the data (called Y row)
            if rowIndexX == rowIndexY:                      # skip comparing a row to itself
                continue
            dataRowY = data.iloc[rowIndexY][1:]             # get the Y row exclusing the ID
            matrix[rowIndexX][rowIndexY] = computeCrossCorrelation(dataRowX, dataRowY)  # add cross correlation to the matrix at position [X, Y]
    print(matrix)
    return matrix


def main():
    data = pd.read_csv("HW_PCA_SHOPPING_CART_v896.csv")
    getCrossCorrMatrix(data)
    

if __name__ == '__main__':
    main()