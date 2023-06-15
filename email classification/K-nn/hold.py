from unicodedata import category
import sklearn.neighbors as skl_nb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold




data= pd.read_csv("mail_data.csv")
data.loc[data['Category'] == 'spam', 'Category',] = 0
data.loc[data['Category'] == 'ham', 'Category',] = 1


X=data["Message"]
Y=data["Category"]

kf = KFold(n_splits=10)
KFold(n_splits=10, random_state=None, shuffle=False)
classification=[]
save_classification=[]
index=[]
for i in range(100):

    for train_index, test_index in kf.split(X):


        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
        X_train_features = feature_extraction.fit_transform(X_train)
        X_test_features = feature_extraction.transform(X_test)
        Y_train = Y_train.astype('int')
        Y_test = Y_test.astype('int')
        model = skl_nb.KNeighborsClassifier(n_neighbors=i+5)
        model.fit(X_train_features,Y_train)
        prediction = model.predict(X_test_features)
        classification.append(np.mean(prediction == Y_test))

 
    save_classification.append(np.mean(classification))
    index=[i]
    print("_______________K-nn________________")
    print(f"Mean accuracy: {np.mean(classification)}")
    print(f"Mean std: {np.std(classification)}")
    print(f"for k = {i+1}")
    

print("______Best run______")
print(f"best mean accuracy {max(save_classification)}")
print(f"for k = {save_classification.index(max(save_classification))+5}")
