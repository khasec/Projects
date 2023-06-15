import sklearn.neighbors as skl_nb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn . ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

data= pd.read_csv("lingspam.csv")

X=data["Body"]
y=data["Label"]

data2= pd.read_csv("mail_data.csv")
data2.loc[data2['Category'] == 'spam', 'Category',] = 0
data2.loc[data2['Category'] == 'ham', 'Category',] = 1

X=X.append(data2["Message"])

y=y.append(data2['Category'])

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)


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


x = "claim your reward here ma buddy"
Input_test = feature_extraction.transform([x])
predictions = model.predict_proba(Input_test)



if model.predict(Input_test) == 1:
    print (f'{x} is not a spam email')
    print(f'this is how sure i am {round(predictions[0][1]*100,1)} %')
else:
    print(f'{x} is a spam email')
    print(f'this is how sure i am {round(predictions[0][0]*100,1)} %')







 
