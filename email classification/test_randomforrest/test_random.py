import sklearn.neighbors as skl_nb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn . ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from mlxtend.evaluate import bias_variance_decomp


data= pd.read_csv("mail_data.csv")
data.loc[data['Category'] == 'spam', 'Category',] = 0
data.loc[data['Category'] == 'ham', 'Category',] = 1


Traindata = np.random.choice(data.shape[0], size=4500, replace=False)
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

model = RandomForestClassifier()


model.fit(X_train_features,Y_train)
classrate=[]
prediction = model.predict(X_test_features)
classrate.append(np.mean(prediction == Y_test))

print(type(prediction))

print(pd.crosstab(prediction,Y_test), '\n')
print(f"Accuracy: {np.mean(prediction == Y_test):.3f}")


#________________________________________________________________________________________________________________________________#

x="lol"



Input_test = feature_extraction.transform([x])



if model.predict(Input_test) == 1:
    print (f'{x} is not a spam email')
    yy = "ham"
else:
    print(f'{x} is a spam email')
    yy = "spam"



#data = {
#    'Category': [yy],
#    'Message': [X_test_newdata],   
#}

# Make data frame of above data
#df = pd.DataFrame(data)
 
# append data frame to CSV file
#df.to_csv('New_data.csv', mode='a', index=False, header=False)














