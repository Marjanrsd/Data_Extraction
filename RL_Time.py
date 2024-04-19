# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:40:28 2024

@author: marjan
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import os
import csv

# This code extracts the total time spent in a maze from csv files

result_dir=r"D:\lizzzz\time-route"
all_csvs= os.listdir(result_dir)
id_time_list=[]

for csv_f in all_csvs:
    id_=csv_f[0:4]
    print(id_)
    data=pd.read_csv(os.path.join(result_dir, csv_f))
    print(data)
    last_row = data.iloc[-1]
    first_row = data.iloc[0]
    end_time = float(last_row[0].replace("Time:", "").strip())
    start_time = float(first_row[0].replace("Time:", "").strip())
    total_time = end_time -  start_time
    id_time_list.append((id_, total_time)) 
        
 
with open("./RL_Time.csv", 'w') as csv_f: 
    csv_w = csv.writer(csv_f, delimiter = ',')   
    csv_w.writerows(id_time_list)

 
    

    

    
    


    



    
            
        
    
    
