import sklearn.neighbors as skl_nb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


#read the data and changes spam to a 0 and not spam to a 1
data= pd.read_csv("mail_data.csv")
data.loc[data['Category'] == 'spam', 'Category',] = 0
data.loc[data['Category'] == 'ham', 'Category',] = 1

#devide the data to train and test data, using around 75% as training data 
Traindata = np.random.choice(data.shape[0], size=4200, replace=False)
trainIndex = data.index.isin(Traindata)
train = data.iloc[trainIndex]
test = data.iloc[~trainIndex]
X_train = train["Message"]
Y_train = train["Category"]
X_test = test["Message"]
Y_test = test["Category"]

# Changes the strings in the data into numerical data (TfidfVectorizer)
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# Inisilasing k-nn method and using k=5
model = skl_nb.KNeighborsClassifier(n_neighbors=1)
model.fit(X_train_features,Y_train)
    
prediction = model.predict(X_test_features)

print(pd.crosstab(prediction,Y_test), '\n')
print(f"Accuracy: {np.mean(prediction == Y_test):.3f}")


#testing a string and displaying the result
x = "Claim your reward"

Input_test = feature_extraction.transform([x])



if model.predict(Input_test) == 1:
    print (f'{x} is not a spam email')
else:
    print(f'{x} is a spam email')