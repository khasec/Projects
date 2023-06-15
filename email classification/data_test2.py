import sklearn.neighbors as skl_nb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn . ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


data= pd.read_csv("New_Data.csv")
""" data.loc[data['label'] == 'spam', 'label',] = 0
data.loc[data['label'] == 'ham', 'label',] = 1 """




from sklearn.model_selection import train_test_split

X=data["Message"]
y=data["Category"]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)






feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
X_train_features = feature_extraction.fit_transform(X_train)

X_test_features = feature_extraction.transform(X_test)

classrate=[]

model = AdaBoostClassifier ( base_estimator = None , n_estimators =100 , learning_rate =1.4)

model.fit(X_train_features,Y_train)
 
prediction = model.predict(X_test_features)
classrate.append(np.mean(prediction == Y_test))


print(pd.crosstab(prediction,Y_test), '\n')
print(f"Accuracy: {np.mean(prediction == Y_test):.3f}")


tdata= pd.read_csv("metamail.csv")
Xt_train = tdata["Message"]
Input_test = feature_extraction.transform(Xt_train)

predictions = model.predict_proba(Input_test)
z=0
for i in predictions:
    print(f'{i[1]} % for message {Xt_train[z]}')
    z=z+1

 












