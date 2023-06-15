import sklearn.neighbors as skl_nb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn . ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from mlxtend.evaluate import bias_variance_decomp
from sklearn.model_selection import train_test_split
from pandas import read_csv

url = 'mail_data.csv'
dataframe = read_csv(url, header=None)
# separate into inputs and outputs
data = dataframe.values

X, y = data[:, -1],data[:, :-1]
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
X = feature_extraction.fit_transform(X)

print(X)

y[0]="ham"
z=0
for i in y:
    if i == "ham":
        y[z]=1
    else:
        y[z]=0
    z=z+1




X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=123,
                                                    shuffle=True)





model = RandomForestClassifier()

avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        model, X_train[1], X_test[1], y_train[1], y_test[1], 
        loss='mse')

print('Average expected loss: %.3f' % avg_expected_loss)
print('Average bias: %.3f' % avg_bias)
print('Average variance: %.3f' % avg_var)