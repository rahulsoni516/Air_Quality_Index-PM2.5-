# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 23:58:43 2020

@author: Rahul
"""

import requests
from bs4 import BeautifulSoup

for year in range(2013,2020):
    for month in range(1,13):
        if(month<10):
            url='https://en.tutiempo.net/climate/0{}-{}/ws-421820.html'.format(month,year)
            
            r=requests.get(url)
            soup=BeautifulSoup(r.text,'html.parser')
            table=soup.find('table',class_='medias mensuales numspan')
            with open ('C:\\Users\\Rahul\\Desktop\\ML_Practice\\AQI','a') as r:
                for tbody in table:
                    for tr in tbody:
                        r.write(tr.text.ljust(8))
                    r.write('\n')
        else:
            url='http://en.tutiempo.net/climate/{}-{}/ws-421820.html'.format(month,year)
            
            r=requests.get(url)
            soup=BeautifulSoup(r.text,'html.parser')
            table=soup.find('table',class_='medias mensuales numspan')
            with open ('C:\\Users\\Rahul\\Desktop\\ML_Practice\\AQI','a') as r:
                for tbody in table:
                    for tr in tbody:
                        r.write(tr.text.ljust(8))
                    r.write('\n')
        


 