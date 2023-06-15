import os
import pandas as pd
import numpy as np

ham_dir = "ham" # Training dataset folder
spam_dir = "spam" # Testing dataset folder

ham_emails = [os.path.join(ham_dir,f) for f in os.listdir(ham_dir)]
spam_emails = [os.path.join(spam_dir,f) for f in os.listdir(spam_dir)]
path = 'hamnspam/'
mails = []
labels = []

for label in ['ham/', 'spam/']:
    filenames = os.listdir(path + label)
    for file in filenames:
        f = open((path + label + file), 'r', encoding = 'latin-1')
        bolk = f.read()
        mails.append(bolk)
        labels.append(label)
        
df = pd.DataFrame({'emails': mails, 'labels': labels})

from sklearn.preprocessing import LabelEncoder
one_coder = LabelEncoder()
df['labels'] = one_coder.fit_transform(df['labels'])

df = df.drop_duplicates()

df['emails'] = df['emails'].apply(lambda x:x.lower())
df['emails'] = df['emails'].apply(lambda x:x.replace('\n', " "))
df['emails'] = df['emails'].apply(lambda x:x.replace('\t', " "))


# Make data frame of above data

 
# append data frame to CSV file
df.to_csv('New_data.csv', mode='a', index=False, header=False)


""" from sklearn.feature_extraction.text import TfidfVectorizer
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')

X = feature_extraction.fit_transform(df['emails']).toarray()

y = df['labels'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test, = train_test_split(X,y,test_size=0.2,random_state=42)


from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier ( base_estimator = None , n_estimators =100 , learning_rate =1.4)

model.fit(X_train,y_train)
 
prediction = model.predict(X_test)



print(pd.crosstab(prediction,y_test), '\n')
print(f"Accuracy: {np.mean(prediction == y_test):.3f}") """