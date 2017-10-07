import matplotlib.pyplot as plt
import numpy as np

X_TrainSet = np.load('/home/giuseppe/ml-project/data/X_train.npy')
print "sono qua"


##############################################################################

'''Import a file from csv'''
import csv
with open('/home/giuseppe/ml-project/data/y_1.csv', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)
print your_list


''' Convert the values of the prediction file'''
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
print predictions

''' Eliminate ALL of the zeros from a picture (Da migliorare)'''

def filterZero(brain):

    filteredBrain = []
    for row in range(0,len(brain)):
        if(brain[row] != 0):
            filteredBrain.append(brain[row])
    return filteredBrain

''' It gives us all of the position where are in the file the old and young brains '''

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

oldBrainPositions = position(predictions,True)
youngBrainPositions = position(predictions,False)

threshold = (min(len(youngBrainPositions),len(oldBrainPositions)))/10





###########################################################TOTAL BRAINS ###########################################################################################################################################################

#######################################################################################################################################################################################################################################



''' albi definition'''

def filterRightSide(brain):

    filteredBrain = []

    for row in range(0,len(brain)):
        if brain[row] > 2500:
            filteredBrain.append(brain[row])
    return filteredBrain


youngBrain = X_TrainSet[0]
oldBrain = X_TrainSet[10]

filteredYoungBrain = filterZero(youngBrain)


filteredOldBrain = filterZero(oldBrain)


minimun = min(min(filteredYoungBrain),min(filteredOldBrain))
maximum = max(max(filteredOldBrain),max(filteredOldBrain))


bins = np.linspace(minimun,maximum,300)

plt.hist(filteredYoungBrain,bins,  alpha = 0.5, label = "20 years old")
plt.hist(filteredOldBrain,bins,  alpha = 0.5, label = "52 years old")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.legend(loc = 'upper right')
plt.title("Brain Comparison")


bins = np.linspace(2500,3663,100)
filteredOldBrainRight = filterRightSide(oldBrain)
filteredYoungBrainRight = filterRightSide(youngBrain)

plt.hist(filteredYoungBrainRight,bins,  alpha = 0.5, label = 'youngBrain')
plt.hist(filteredOldBrainRight,bins,  alpha = 0.5, label = 'oldBrain')
plt.legend(loc = 'upper right')
plt.title("brain comparison")


