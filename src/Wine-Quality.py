"""
Prediction of wine quality using machine learning in python

"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split as tts
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from array import array

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Build the path to the dataset file
dataset_path = os.path.join(current_dir, '../dataset/WineQuality-White.csv')

# reading dataset
raw_data = pd.read_csv(dataset_path,sep=';')
print("Shape of raw data :",raw_data.shape,"\n")

dup_data = raw_data.duplicated()
print("Number of duplicate rows =", sum(dup_data),"\n")

data = raw_data.drop_duplicates()
print("Shape of data after removing duplicates :",data.shape,"\n")

data.rename(columns = {'fixed acidity':'fixed_acidity','volatile acidity':'volatile_acidity',\
                     'citric acid':'citric_acid','residual sugar':'residual_sugar',\
                     'free sulfur dioxide':'free_sulfur_dioxide',\
                     'total sulfur dioxide':'total_sulfur_dioxide'},inplace = True)

# printing first 5 data instances
print(data.head())

print(data.isnull().sum())

# data set description
data_info=data.describe()
print(data_info)

data_corr=data.corr()
print(data_corr)

# splitting the data set
train,test=tts(data,test_size=0.2)

y = train['quality']
feature_names = ["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides",\
      "free_sulfur_dioxide","total_sulfur_dioxide","pH","sulphates","alcohol"]
X = train[feature_names]

# model formation
reg=linear_model.LinearRegression()
model=reg.fit(X,y)

print("Coefficients of the linear equation : \n",reg.coef_,"\n")
print("Y-intercept :",reg.intercept_)

y_train_pred = reg.predict(X)
print("In sample Root mean square error: %.2f"%mean_squared_error(y,y_train_pred)**0.5)

y_test = test['quality']
X_test=test[feature_names]

y_test_pred = reg.predict(X_test)
print("Out sample Root mean square error: %.2f"%mean_squared_error(y_test,y_test_pred)**0.5)

# unknown sample
a=np.array([12,0.5,1.4,56,0.2,205,380,3.1,0.3,7.1]).reshape(1,-1)
a_df = pd.DataFrame(a, columns=feature_names)
quality1=reg.predict(a_df)
print("Quality of predefined wine sample is:", quality1)


# unknown sample from user
arr = array('f',[])
x = float(input("Enter the value of fixed acidity(range(3 to 15))"))
arr.append(x)
x=float(input("Enter the value of volatile acidity(range(0 to 1))"))
arr.append(x)
x=float(input("Enter the value of citric acidity(range(0 to 2))"))
arr.append(x)
x=float(input("Enter the value of residual sugar(range(0 to 100))"))
arr.append(x)
x=float(input("Enter the value of chlorides(range(0 to 0.5))"))
arr.append(x)
x=float(input("Enter the value of free sulphur dioxide(range(0 to 300))"))
arr.append(x)
x=float(input("Enter the value of total sulphur dioxide(range(0 to 500))"))
arr.append(x)
x=float(input("Enter the value of pH(range(2 to 4))"))
arr.append(x)
x=float(input("Enter the value of sulphates(range(0 to 1))"))
arr.append(x)
x=float(input("Enter the value of alcohol(range(5 to 15))"))
arr.append(x)


ar=np.asarray(arr)
ar=ar.reshape(1,-1)
ar_df = pd.DataFrame(ar, columns=feature_names)
quality2 = reg.predict(ar_df)

print("Quality of your wine sample is",quality2)
