import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv(r"hiring.csv")
print(dataset)

# null value handling
dataset['experience'].fillna(0,inplace=True)
dataset['test_score'].fillna(dataset['test_score'].mean(),inplace=True)

def encode_word(word):
    word_dic ={0:0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11}
    return word_dic[word]

# change the experience column from categorical to numerical
dataset['experience'] = dataset['experience'].apply(lambda x :  encode_word(x))
print(dataset)

# extract the predictor variables and target variables
X = dataset.iloc[:,0:3]
Y = dataset.iloc[:,-1]

# modelling
model = LinearRegression()
model.fit(X,Y)
print("Completed model training")

print("Training Score:" + str(model.score(X,Y)))

print(model.predict([[1,8,9]]))
#Saving the model using joblib
joblib.dump(model,"hiring_model.pkl")
