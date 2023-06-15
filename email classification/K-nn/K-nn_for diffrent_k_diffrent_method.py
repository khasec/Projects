import sklearn.neighbors as skl_nb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

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
X_test = train["Message"]
Y_test = train["Category"]

# Changes the strings in the data into numerical data (CountVectorizer method)
cv = CountVectorizer()
X_train_features = cv.fit_transform(X_train)
X_test_features = cv.fit_transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


classrate=[]
#running the test for 100 diffrent k 
for i in range(100):
    model = skl_nb.KNeighborsClassifier(n_neighbors=i+1)
    model.fit(X_train_features,Y_train)
    print(i)
    prediction = model.predict(X_test_features)
    classrate.append(np.mean(prediction == Y_test))




#plot the result 
i=np.linspace(1,100,100)
plt.plot(i,classrate)
plt.show()

print(pd.crosstab(prediction,Y_test), '\n')
print(f"Accuracy: {np.mean(prediction == Y_test):.3f}")
