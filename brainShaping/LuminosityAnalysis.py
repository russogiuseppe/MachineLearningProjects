import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
X_TrainSet = np.load('/home/giuseppe/ml-project/data/X_train.npy')
print "sono qua"

'''Import the prediction.csv'''
import csv
with open('/home/giuseppe/ml-project/data/y_1.csv', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)

'''Caricamento da file dei vari parametri
    1) PARAMETER = sottoinsieme di cervelli su cui costruire le nostre features 
    2) SIZE = dimensione dell'array singleArrayLuminosity puoi vedere sotto
'''


''' Converting the values of the prediction file into a single list of integers'''
def csvList(list):

    counter = len(list)
    predictions = []

    for i in range(0,counter):
        tmp = len(list[i])
        for j in range(0,tmp):
            value = int(list[i][j])
            predictions.append(value)
    return predictions

predictions = csvList(your_list)

''' Eliminate ALL of the zeros from a picture (Da migliorare)'''

'''Se vedi che non ho implementato il codice con l'algoritmo che ti dicevo fallo tu
Significa che lo ho dimenticato'''


def filterZero(brain):

    filteredBrain = []
    for row in range(0,len(brain)):
        if(brain[row] != 0):
            filteredBrain.append(brain[row])
    return filteredBrain

''' It gives us all of the position where are in the file the old and young brains '''
'''Not usefull anymore
def position(predictions,old):

    counter = len(predictions)
    vector = []
    if(old):
        for i in range(0,counter):
            if(predictions[i] >= 70):
                vector.append(i)
        return vector
    else:
        for i in range(0,counter):
            if(predictions[i] <= 30):
                vector.append(i)
        return vector

youngBrainPositions = position(predictions,False)
oldBrainPositions = position(predictions,True)

threshold = min(len(youngBrainPositions),len(oldBrainPositions))
'''

'''Work out the highest intensity within the dataset'''

def maxLuminosity(trainSet):

    tmp = []
    for i in range(0,len(trainSet)):
        tmp.append(max(trainSet[i]))
    return max(tmp)

''' Luminosity of one of the brains'''

''' Scrivi su file dio cane'''
size = maxLuminosity(X_TrainSet)


''' This creates the histogram values within the array'''


def singleArrayLuminosity(brain,size):

    filteredBrain = filterZero(brain)
    lst = [0]*size
    for counter in range(0,len(filteredBrain)):
        lst[filteredBrain[counter]] = lst[filteredBrain[counter]] + 1
    return lst


''' Don't use this: It could be usefull, but not now'''

def arraysum(array,i,j):

    sum = 0
    for i in range(i,j):
       sum = sum + array[i]
    return sum

PARAMETER = 200
#lowerBoundary = 700
#upperBoundary = 800

''' Features extraction, luminosity values'''

def features(trainSet,start, end):
    features = []
    for i in range(start, end):
        tmp = singleArrayLuminosity(trainSet[i],size)
        features.append(tmp)
    return features


''' First attempt'''
'''I am gonna create a dataframe which has as rows the brains 
as columns the different intensity values'''


trainSet = features(X_TrainSet, 0, PARAMETER)

validationSet = features(X_TrainSet, PARAMETER, len(X_TrainSet))
'''
firstAttemptFeature = open('firstAttempt.txt','w')
for item in firstAttempt:
    firstAttemptFeature.write("%s\n" %item)
firstAttemptFeature.close()
'''
######################################################################################################################

df = pd.DataFrame(trainSet)

validationDataFrame = pd.DataFrame(validationSet)

df.to_csv('finalFeatures.csv')
validationDataFrame.to_csv('validationSet.csv')

'''''''''''''''''''''''''''''''''''''''''''''''''''END OF FEATURE EXTRACTIONS'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

regr = linear_model.LinearRegression()

#train the model

regr.fit(trainSet,predictions)

# a little of cross validation, using as test set the lst part of the csv

validationPrediction = regr.predict(validationSet)

print validationPrediction[0]




##################################################################################################################################################
'''Non sappiamo se ci serve è quella che abbiamo discusso ieri con Simon
    Possibilmente è inutile, in ogni caso è salvata su file'''
'''
def feature1(trainSet,threshold):

    feature1 = []
    for index in range(0,threshold):
        luminosity = singleArrayLuminosity(X_TrainSet[index],size)
        elem = arraysum(luminosity,lowerBoundary,upperBoundary)
        feature1.append(elem)
    return feature1

feature1 = feature1(X_TrainSet,PARAMETER)
'''

'''Salvo su file feature 1'''

'''
Feature1 = open('feature1.txt','w')
for item in feature1:
    Feature1.write("%d\n" %item)
Feature1.close()
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''
def variance(trainSet,parameter):


    variance = []
    for index in range(0,parameter):
        luminosity = singleArrayLuminosity(trainSet[index],size)
        luminosity = filterZero(luminosity)
        variance.append(np.var(luminosity))
    return variance

feature2 = variance(X_TrainSet,PARAMETER)
Feature2 = open('feature2.txt','w')
for item in feature1:
    Feature2.write("%d\n" %item)
Feature2.close()


def limit(array,paramenter):
    final = []
    for elem in range(0,paramenter):
        final.append(array[elem])
    return final

trainPredictions = limit(predictions,PARAMETER)

print len(feature1)
print len(trainPredictions)
p = plt.plot(feature1,trainPredictions,"o")
s = plt.plot(feature2,trainPredictions,"o")
'''
'''
luminosity0 = singleArrayLuminosity(X_TrainSet[0],size)
luminosity1 = singleArrayLuminosity(X_TrainSet[3],size)

luminosity2 = singleArrayLuminosity(X_TrainSet[1],size)

'''


'''

value0 = arraysum(luminosity0,700,800)
value1 = arraysum(luminosity1,700,800)
value2 = arraysum(luminosity2,700,800)


difference = value0/value1

difference = value0/value2
'''