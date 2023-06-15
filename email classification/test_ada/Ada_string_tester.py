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
x = "Go to the website, www.freemoney.nu"
Input_test = feature_extraction.transform([x])

predictions = model.predict_proba(Input_test)



if model.predict(Input_test) == 1:
    print (f'{x} is NOT a spam email')
    print(f'this is how sure i am {round(predictions[0][1]*100,1)} %')
else:
    print(f'{x} is A spam email')
    print(f'this is how sure i am {round(predictions[0][0]*100,1)} %')



""" data = {
    'Category': [yy],
    'Message': [x],   
}

# Make data frame of above data
df = pd.DataFrame(data)
 
# append data frame to CSV file
df.to_csv('New_data.csv', mode='a', index=False, header=False) """