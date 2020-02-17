# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:49:50 2020

@author: Rahul
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("C:\\Users\\Rahul\\Desktop\\ML_Practice\\Rahul_AQI\\final.csv")

#checking for null values
import seaborn as sns
sns.heatmap(data.isnull(),yticklabels= False,cbar=True,cmap='viridis')

data=data.dropna()

x=data[['T','SLP','VV','V','VM']]
#x=data.iloc[:, :-1] ##independent features
y=data.iloc[:,-1]  ##dependent features

corr_mat=data.corr()
top_corr_features=corr_mat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

Covariance=data.cov()

sns.pairplot(data)

##Feature Importance
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(x,y)
print(model.feature_importances_)

importance_feature=pd.Series(model.feature_importances_,index=x.columns)
importance_feature.nlargest(5).plot(kind='bar')
plt.show()

sns.distplot(y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

from sklearn.model_selection import cross_val_score
score=cross_val_score(reg,x,y,cv=8)
score.mean()

reg.coef_
reg.intercept_

df=pd.DataFrame(reg.coef_,x.columns,columns=['Coefficient'])
predict=reg.predict(x_test)
sns.distplot(predict-y_test)

print("Coff. of R^2: ", reg.score(x_train,y_train))
print("Coff. of R^2: ", reg.score(x_test,y_test))

##Model Evaluation
from sklearn import metrics
print("MAE : ", metrics.mean_absolute_error(predict,y_test))
print("MSE : ",metrics.mean_squared_error(predict,y_test))
print("RMSE: ",np.sqrt(metrics.mean_squared_error(predict,y_test)))

'''
MAE :  51.48421871271269
MSE :  4399.725836581429
RMSE:  66.33042919039066
'''

=================================================================
#comparision Linear,Ridge & Lasso
##Linear Regression
MSE=cross_val_score(reg, x,y,scoring='neg_mean_squared_error',cv=10)
print(np.mean(MSE))

##Ridge regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge=Ridge()
parameters={'alpha':['1e-15','1e-10','1e-5','1e-2','1e-1',1,5,10,20,30]}
ridge_reg=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_reg.fit(x,y)

print(ridge_reg.best_params_)
print(ridge_reg.best_score_)

##Lasso Regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
 
parameters={'alpha':[1,5,10,20,30]}
lasso_reg=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_reg.fit(x,y)

print(lasso_reg.best_params_)
print(lasso_reg.best_score_)


'''
#Linear
-4906.630908338663
#Ridge
{'alpha': 30}
-4882.97779608358
#Lasso
{'alpha': 5}
-4863.443851852203
'''

predict=lasso_reg.predict(x_test)
sns.distplot(predict-y_test)

from sklearn import metrics
print("MAE : ", metrics.mean_absolute_error(predict,y_test))
print("MSE : ",metrics.mean_squared_error(predict,y_test))
print("RMSE: ",np.sqrt(metrics.mean_squared_error(predict,y_test)))


'''
MAE :  51.98684388788675
MSE :  4505.88751106091
RMSE:  67.12590789747956
''' 

y_temp=y_test.values

plt.plot(predict,label="predict")
plt.plot(y_temp,label="y_test")
plt.xlabel('Day')
plt.ylabel('PM 2.5')
plt.legend(loc='top center')
plt.show()

plt.scatter(y_test,predict)
plt.scatter(y_temp,predict)
