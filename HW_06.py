import pandas as pd
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
    avgS = getAverage(s)
    avgT = getAverage(t)

    stdDevS = statistics.stdev(s)
    stdDevT = statistics.stdev(t)

    crossCorrSum = 0
    for k in range(0, N):
        crossCorrSum += ((s[k] - avgS)/stdDevS) * ((t[k] - avgT)/stdDevT)

    return crossCorrSum/N


def main():
    data = pd.read_csv("HW_PCA_SHOPPING_CART_v896.csv")
    # print(data)
    sampleS = [1, 2, 3, 4, 5, 6, 7]
    sampleT = [14, 13, 12, 11, 10, 9, 8]
    print(computeCrossCorrelation(sampleS, sampleT))

if __name__ == '__main__':
    main()