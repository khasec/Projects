# estimate the bias and variance for a regression model
from pandas import read_csv
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from mlxtend.evaluate import bias_variance_decomp
import matplotlib.pyplot as plt

# load dataset
url = 'mail_data.csv'
dataframe = read_csv(url, header=None)
# separate into inputs and outputs
data = dataframe.values

X, y = data[:, -1],data[:, :-1]
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
X = feature_extraction.fit_transform(X)


X=X.toarray()
y[0]="ham"
z=0
for i in y:
    if i == "ham":
        y[z]=1
    else:
        y[z]=0
    z=z+1



model = DecisionTreeClassifier()
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        model, X, y, X, y, 
        loss='mse',
        random_seed=123)

print('Average expected loss: %.3f' % avg_expected_loss)
print('Average bias: %.3f' % avg_bias)
print('Average variance: %.3f' % avg_var)


