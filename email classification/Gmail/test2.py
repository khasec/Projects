


import pandas as pd

data= pd.read_csv("email.csv") 

X=data["Message"]
Y=data["Category"]

z=0
for i in X:
    
    data = {
    'Category': [Y[z]],
    'Message': [i],
    'real':[Y[z]],   
    }
    df = pd.DataFrame(data)
    
    df.to_csv("mail.csv", mode='a', index=False, header=False)
    z=z+1

    