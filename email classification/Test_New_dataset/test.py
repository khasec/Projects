import sklearn.neighbors as skl_nb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn . ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.evaluate import bias_variance_decomp
from sklearn.model_selection import train_test_split

#read the data and changes spam to a 0 and not spam to a 1
data= pd.read_csv("New_data.csv")


# Deviding the data in to test data and traning data, around 75% is training data




X=data["Message"]
y=data["Category"]


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)


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






