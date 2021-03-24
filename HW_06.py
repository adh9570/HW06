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


def main():
    data = pd.read_csv("HW_PCA_SHOPPING_CART_v896.csv")
    # data = pd.read_csv("sample data.csv")
    # getCrossCorrMatrix(data)
    columnsCrossCorr(data)
    

if __name__ == '__main__':
    main()