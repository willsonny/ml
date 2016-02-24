import csv
import math
import random
import operator


def loadDataset(filename, split, trainDataset=[], testDataSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataSet = list(lines)
        for x in range(len(dataSet) - 1):
            for y in range(4):
                dataSet[x][y] = float(dataSet[x][y])
            if random.random() < split:
                trainDataset.append(dataSet[x])
            else:
                testDataSet.append(dataSet[x])


def euclideanDistance(instance1, instace2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instace2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainDataset, testInstance, k):
    distance = []
    length = len(testInstance) - 1
    for x in range(len(trainDataset)):
        dist = euclideanDistance(testInstance, trainDataset[x], length)
        distance.append((trainDataset[x], dist))
    distance.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distance[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100


def main():
    trainDataset = []
    testDataset = []
    split = 0.67
    loadDataset(r'iris.txt', split, trainDataset, testDataset)
    print 'Train set:' + repr(len(trainDataset))
    print 'Test set:' + repr(len(testDataset))

    predictions = []
    k = 3
    for x in range(len(testDataset)):
        neighbors = getNeighbors(trainDataset, testDataset[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> prediction = ' + repr(result) + 'actual = ' + repr(testDataset[x][-1]))
    accuracy = getAccuracy(testDataset, predictions)
    print('Accuracy : ' + repr(accuracy) + '%')


main()
