# -*- coding: utf-8 -*-

# Predict the price of the stock of a company

"""## [Step 1] Import basic libraries"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import os

STUDENT_ID = "12345678" ##Make sure to update this for your code to be graded

"""## [Step 2] Loading the dataset"""

df_train = pd.read_csv('train.csv')['price'].values
df_train = df_train.reshape(-1, 1)
df_test = pd.read_csv('test.csv')['price'].values
df_test = df_test.reshape(-1, 1)

dataset_train = np.array(df_train)
dataset_test = np.array(df_test)

"""# [Step 3] Pre process your data (no restrictions) """

## Pre process your data in any way you want
scaler = None
##########################################

"""### We create the X_train and Y_train from the dataset train
We take a price on a date as y_train and save the previous 50 closing prices as x_train
"""

trace_back = 50
def create_dataset(df):
    x, y = [], []
    for i in range(trace_back, len(df)):
        x.append(df[i-trace_back:i, 0])
        y.append(df[i, 0])
    return np.array(x),np.array(y)

x_train, y_train = create_dataset(dataset_train)

x_test, y_test = create_dataset(dataset_test)

"""## [Step 4] Build your RNN model
### You are expect to change the content in the below cell and add your own cells

1. You have to design a RNN model that takes in your x_train and do prediction on x_test
2. Your model should be able to predict on x_test using model.predict(x_test)
3. Do not use any pretrained model.
"""

## Your RNN model goes here
model = None
##########################################

"""## [Step 5]: Predictions on X_test
DO NOT change the below code
"""

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

"""## [Step 6]: Checking the Root Mean Square Error on X_test"""

rmse_score = mean_squared_error([x[0] for x in y_test_scaled], [x[0] for x in predictions], squared=False)
print("RMSE",STUDENT_ID,":",rmse_score)
