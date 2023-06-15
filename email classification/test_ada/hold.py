import sklearn.neighbors as skl_nb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn . ensemble import AdaBoostClassifier


data= pd.read_csv("mail_data.csv")
data.loc[data['Category'] == 'spam', 'Category',] = 0
data.loc[data['Category'] == 'ham', 'Category',] = 1



Traindata = np.random.choice(data.shape[0], size=4000, replace=False)
trainIndex = data.index.isin(Traindata)
train = data.iloc[trainIndex]
test = data.iloc[~trainIndex]
X_train = train["Message"]
Y_train = train["Category"]
X_test = train["Message"]
Y_test = train["Category"]
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


x = "January only! Had your mobile 12mths+? You are entitled to update to the latest camera mobile for Free Free! Call The Mobile Update Co FREE on 8121291126"
Input_test = feature_extraction.transform([x])



if model.predict(Input_test) == 1:
    print (f'{x} is NOT a spam email')
else:
    print(f'{x} is A spam email')
