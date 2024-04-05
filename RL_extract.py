# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:40:05 2023

@author: marja
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import imageio
import os
import csv
import glob


result_dir=r"D:\lizzzz\Route Learning2"
all_csvs= os.listdir(result_dir)
id_score_list=[]

for csv_f in all_csvs:
    id_=csv_f[0:4]
    print(id_)
    data=pd.read_csv(os.path.join(result_dir, csv_f))
    print(data)
    num_err=data.shape[0]
    id_score_list.append((id_, num_err)) 
        
 
with open("./RL2.csv", 'w') as csv_f: 
    csv_w = csv.writer(csv_f, delimiter = ',')   
    csv_w.writerows(id_score_list)

 
    

    

    
    


    



    
            
        
    
    
