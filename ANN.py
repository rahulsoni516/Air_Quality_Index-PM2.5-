# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 08:24:09 2020

@author: Rahul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

data=pd.read_csv("C:\\Users\\Rahul\\Desktop\\ML_Practice\\Rahul_AQI\\final.csv")
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data=data.dropna()

    
x=data.iloc[:, :-1] ##independent features
y=data.iloc[:,-1]  ##dependent features

sns.distplot(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

model = Sequential()

# The Input Layer :
model.add(Dense(8, kernel_initializer='normal',input_dim = x_train.shape[1], activation='relu'))

# The Hidden Layers :
model.add(Dense(16, kernel_initializer='normal',activation='relu'))
model.add(Dense(16, kernel_initializer='normal',activation='relu'))
model.add(Dense(16, kernel_initializer='normal',activation='relu'))

# The Output Layer :
model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
model.summary()

# Fitting the ANN to the Training set
model_history=model.fit(x_train, y_train,validation_split=0.33, batch_size = 10, nb_epoch = 100)

#Model Evaluation
y_pred=model.predict(x_test)

sns.distplot(y_test.values.reshape(-1,1)-y_pred)
plt.scatter(y_test,y_pred)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

'''
MAE: 49.705328746898246
MSE: 4223.253826996874
RMSE: 64.98656651183283
'''


y_temp=y_test.values

plt.plot(y_pred,label="predicted PM2.5")
plt.plot(y_temp,label="Actual PM2.5")
plt.xlabel('Day')
plt.ylabel('PM 2.5')
plt.legend(loc='top center')
plt.show()









