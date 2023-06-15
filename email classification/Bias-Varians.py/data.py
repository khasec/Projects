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
mail = data[:, -1]
spamham = data[:, :-1]

listspam=[]
z=0
for i in mail:
    if spamham[z] == "spam":
        listspam.append(i)
    z=z+1
string_data_spam=np.char.split(listspam)

dic={}
for i in string_data_spam:
    for z in i:
        if z in dic:
            dic[z] = dic[z] + 1 
            
        else:
            dic[z] = 1

sorted_dict = {}
sorted_keys = sorted(dic, key=dic.get)

for w in sorted_keys:
    sorted_dict[w] = dic[w]

print(sorted_dict)
