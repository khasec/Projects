from sklearn import preprocessing
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
def derivative(x):
    return x * (1.0 - x)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
X = []
Y = []
# read the training data
data= pd.read_csv("mail_data.csv")
data.loc[data['Category'] == 'spam', 'Category',] = 0
data.loc[data['Category'] == 'ham', 'Category',] = 1
X=data["Message"]
Y=data["Category"]
X = np.array(X)

Y = np.array(Y)
# the first 2500 out of 3000 emails will serve as training data
X_train = X[0:2500]
y_train = Y[0:2500]
# the rest 500 emails will serve as testing data
X_test = X[2500:]
y_test = Y[2500:]
X = X_train
y = y_train
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
X_train = feature_extraction.fit_transform(X_train)
X_test = feature_extraction.transform(X_test)
y_train = y_train.astype('int')
y_test = y_test.astype('int')
X_train = preprocessing.scale(X_train) # feature scaling
X_test = preprocessing.scale(X_test) # feature scaling
# we have 3 layers: input layer, hidden layer and output layer
# input layer has 57 nodes (1 for each feature)
# hidden layer has 4 nodes
# output layer has 1 node
dim1 = len(X_train[0])
dim2 = 4
# randomly initialize the weight vectors
np.random.seed(1)
weight0 = 2 * np.random.random((dim1, dim2)) - 1
weight1 = 2 * np.random.random((dim2, 1)) - 1
# you can change the number of iterations
for j in xrange(25000):
    # first evaluate the output for each training email
    layer_0 = X_train
    layer_1 = sigmoid(np.dot(layer_0,weight0))
    layer_2 = sigmoid(np.dot(layer_1,weight1))
    # calculate the error
    layer_2_error = Y_train - layer_2
    # perform back propagation
    layer_2_delta = layer_2_error * derivative(layer_2)
    layer_1_error = layer_2_delta.dot(weight1.T)
    layer_1_delta = layer_1_error * derivative(layer_1)
    # update the weight vectors
    weight1 += layer_1.T.dot(layer_2_delta)
    weight0 += layer_0.T.dot(layer_1_delta)
# evaluation on the testing data
layer_0 = X_test
layer_1 = sigmoid(np.dot(layer_0,weight0))
layer_2 = sigmoid(np.dot(layer_1,weight1))
correct = 0
# if the output is > 0.5, then label as spam else no spam
for i in xrange(len(layer_2)):
    if(layer_2[i][0] > 0.5):
        layer_2[i][0] = 1
    else:
        layer_2[i][0] = 0
    if(layer_2[i][0] == y_test[i][0]):
        correct += 1
# printing the output
