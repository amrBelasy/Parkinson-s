#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: amr
"""

import pandas as pd
import os, sys

# read the Data
df = pd.read_csv('parkinsons.data') 

df.head()

# Get the features and labels
features = df.loc[:,df.columns != 'status'].values[:,1:]
labels = df.loc[:,'status'].values

# Get the count of each label (0 and 1) in labels
print(labels[labels == 1].shape[0], labels[labels == 0].shape[0])

# Scale the features to between -1 and 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((-1,1))
x = scaler.fit_transform(features)
y = labels

# Split the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)


# Train the model
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_train, y_train)

# Calculate the accuracy
from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)

