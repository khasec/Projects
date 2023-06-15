from unicodedata import category
import sklearn.neighbors as skl_nb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split



#read the data and changes spam to a 0 and not spam to a 1
df= pd.read_csv("spambase_csv.csv")


scaler = StandardScaler()
scaled_df = pd.DataFrame(data = scaler.fit_transform(df.drop('class',axis=1)), columns=df[df.drop('class', axis=1).columns].columns)
scaled_df.head()

X = scaled_df
Y = df['class']



X_train, X_test, Y_train , Y_test = train_test_split(X, Y ,test_size=0.33, random_state=42)


base = DecisionTreeClassifier(max_depth=1)
model = AdaBoostClassifier ( base_estimator = None , n_estimators =170 , learning_rate =1.3)
model.fit(X_train,Y_train)
prediction = model.predict(X_test)

print(pd.crosstab(prediction,Y_test), '\n')
print(f"Accuracy: {np.mean(prediction == Y_test):.3f}")

 
    


    
    
