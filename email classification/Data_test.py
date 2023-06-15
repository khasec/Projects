import sklearn.neighbors as skl_nb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn . ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import os


""" data= pd.read_csv("spam_ham_dataset.csv")
data.loc[data['label'] == 'spam', 'label',] = 0
data.loc[data['label'] == 'ham', 'label',] = 1

Traindata = np.random.choice(data.shape[0], size=4500, replace=False)
trainIndex = data.index.isin(Traindata)
train = data.iloc[trainIndex]
test = data.iloc[~trainIndex]
X_train = train["text"]
Y_train = train["label"]
X_test = train["text"]
Y_test = train["label"] """

data= pd.read_csv("mail_data.csv")
data.loc[data['Category'] == 'spam', 'Category',] = 0
data.loc[data['Category'] == 'ham', 'Category',] = 1


Traindata = np.random.choice(data.shape[0], size=2000, replace=False)
trainIndex = data.index.isin(Traindata)
train = data.iloc[trainIndex]
test = data.iloc[~trainIndex]
X_train = train["Message"]
Y_train = train["Category"]
X_test = test["Message"]
Y_test = test["Category"]



feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
X_train_features = feature_extraction.fit_transform(X_train)

X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')



classrate=[]

model = AdaBoostClassifier ( base_estimator = None , n_estimators =100 , learning_rate =1.4)

model.fit(X_train_features,Y_train)
 
prediction = model.predict(X_test_features)
classrate.append(np.mean(prediction == Y_test))


print(pd.crosstab(prediction,Y_test), '\n')
print(f"Accuracy: {np.mean(prediction == Y_test):.3f}")


x = "hello isak :)"
Input_test = feature_extraction.transform([x])
predictions = model.predict_proba(Input_test)

print(Input_test)

if model.predict(Input_test) == 1:
    print (f'{x} is not a spam email')
    print(f'this is how sure i am {round(predictions[0][1]*100,1)} %')
else:
    print(f'{x} is a spam email')
    print(f'this is how sure i am {round(predictions[0][0]*100,1)} %')

 












