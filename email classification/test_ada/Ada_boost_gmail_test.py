import sklearn.neighbors as skl_nb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn . ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.evaluate import bias_variance_decomp

#read the data and changes spam to a 0 and not spam to a 1
data= pd.read_csv("mail_data.csv")
data.loc[data['Category'] == 'spam', 'Category',] = 0
data.loc[data['Category'] == 'ham', 'Category',] = 1

# Deviding the data in to test data and traning data, around 75% is training data


train = data

X_train = train["Message"]
Y_train = train["Category"]


# Changes the strings in the data into numerical data
feature_extraction = TfidfVectorizer(min_df = 1, lowercase='False')
X_train_features = feature_extraction.fit_transform(X_train)


Y_train = Y_train.astype('int')


# Inisilasing Adaptive Boosting with base_estimator = base , n_estimators =100 , learning_rate =1.2
classrate=[]
base = DecisionTreeClassifier(max_depth=1)
model = AdaBoostClassifier ( base_estimator = base , n_estimators =100, learning_rate =1.2)


model.fit(X_train_features,Y_train)
 


#input test string and displaying the result. // claim

#test data:
tdata= pd.read_csv("metamail.csv")
Xt_train = tdata["Message"]

Input_test = feature_extraction.transform(Xt_train)

predictions = model.predict_proba(Input_test)
z=0
#for i in predictions:
   # print(f'{i[1]} % for message {Xt_train}')
   # z=z+1





