''' Import il csv con le features'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

'''Import the prediction.csv'''
import csv
with open('finalFeatures.csv', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)


'''Import the prediction.csv'''
import csv
with open('validationSet.csv', 'rb') as f:
    reader = csv.reader(f)
    validation = list(reader)

import csv
with open('/home/giuseppe/ml-project/data/y_1.csv', 'rb') as f:
    reader = csv.reader(f)
    stringPrediction = list(reader)


def csvList(list):

    counter = len(list)
    predictions = []

    for i in range(0,counter):
        tmp = len(list[i])
        for j in range(0,tmp):
            value = int(list[i][j])
            predictions.append(value)
    return predictions

total = csvList(stringPrediction)
def dataframeConversion(dataframe):

    features = []
    for row in range(1,len(dataframe)):
        size = len(dataframe[row])
        tmpList = []
        for column in range(1,size):
            if(dataframe[row][column] == ''):
                value = 0
                tmpList.append(value)
            else:
                value = int(dataframe[row][column])
                tmpList.append(value)
        features.append(tmpList)
    return features

def limit(array,paramenter):
    final = []
    for elem in range(0,paramenter):
        final.append(array[elem])
    return final

prediction = csvList(stringPrediction)
predictions = limit(prediction,200)
trainValues = dataframeConversion(your_list)
validationSet = dataframeConversion(validation)

def select(array, start):
    final = []
    for index in range(start,len(array)):
        final.append(array[index])
    return final


crossValidateSet = select(total,200)


print len(validationSet)
'''Regression '''

'''Linear Regression'''

regr = linear_model.LinearRegression()

#train the model
print len(trainValues)

regr.fit(trainValues,predictions)

# a little of cross validation, using as test set the lst part of the csv

validationPrediction = regr.predict(validationSet)
firstPrediction = pd.DataFrame(validationPrediction)
firstPrediction.to_csv('firstPrediction.csv')
print validationPrediction[2]


def error(prediction,fakeprediction):

    sum = 0
    for i in range(0,len(prediction)):

        value = (prediction[i] - fakeprediction[i])**2
        sum = sum + value
    final = sum/len(prediction)
    return final

result = error(crossValidateSet,validationPrediction)
