# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 21:51:50 2020

@author: Rahul
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

temp_i=0
average=[]

for year in range(2013,2020):    
    for rows in pd.read_csv('C:/Users/Rahul/Desktop/ML_Practice/AQI_Project/Data/aqi{}.csv'.format(year),chunksize=24):
        add_var=0
        avg=0.0
        data=[]
        df=pd.DataFrame(data=rows)
        for index,row in df.iterrows():
            data.append(row['PM2.5'])
        for i in data:
            if type(i) is float or type(i) is int:
                add_var=add_var+i
            elif type(i) is str:
                if i!='NoData' and i!='PwrFail' and i!='---' and i!='InVld':
                    temp=float(i)
                    add_var=add_var+temp
        avg=add_var/24
        temp_i=temp_i+1
        
        average.append(avg)
       
type(average)
d1=pd.DataFrame(average)
d1.columns = ['PM2.5']

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=0, strategy='mean')
d1=imp.fit_transform(d1)


d1=pd.DataFrame(d1)


plt.plot(average,label="AQI 2013-2019")
plt.xlabel('Day')
plt.ylabel('PM 2.5')
plt.legend(loc='top center')
plt.show()

=============================================================================================

data=pd.read_excel("C:\\Users\\Rahul\\Desktop\\ML_Practice\\new_data.xlsx", encoding='unicode_escape')
#data.dtypes
tem_data=data[['T','TM','Tm','SLP','H','VV','V','VM']]

tem_data=tem_data.replace('-',np.nan)

#from sklearn.impute import SimpleImputer
#imp = SimpleImputer(missing_values='-', strategy='mean')
#tem_data=imp.fit_transform(tem_data)

semifinal=pd.concat([tem_data, d1], axis=1, sort=False)

#droping rows with null values for all features

#final_data = semifinal[np.isfinite(semifinal['T'])]

final=semifinal.dropna(subset=['T']) 
final.to_csv("C:\\Users\\Rahul\\Desktop\\ML_Practice\\AQI_Project\\final.csv",index = None, header=True)
#from sklearn.impute import SimpleImputer
#imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#final=imp.fit_transform(final)









