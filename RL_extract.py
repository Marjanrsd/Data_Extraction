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


# Read multiple txt files from a directory- first method
# path=r"D:\lizzzz\Route Learning"
# os.chdir(path)
# def read_text_file(file_path):
#     with open (file_path, "r") as f:
#         print(f.read())
        
# for file in os.listdir():
#     if file.endswith(".txt"):
#         file_path=f"{path}\{file}"
#         read_text_file(file_path)

# # Read files using pandas- second method
# folder_path=r"D:\lizzzz\Route Learning"
# file_list=glob.glob(folder_path+"/*.txt")
# main_dataframe = pd.DataFrame(pd.read_csv(file_list[0]))
# for i in range(1, len(file_list)):
#     data=pd.read_csv(file_list[i])
#     df=pd.DataFrame(data)
#     main_dataframe = pd.concat([main_dataframe,df],axis=1) 
# print(main_dataframe)
    

# Read files 3rd method
result_dir=r"D:\lizzzz\Route Learning2"
all_csvs= os.listdir(result_dir)
id_score_list=[]

for csv_f in all_csvs:
    id_=csv_f[0:4]
    print(id_)
    # for i in range(len(all_csvs)):
    data=pd.read_csv(os.path.join(result_dir, csv_f))
    num_err=data.count()
    print(num_err)
    id_score_list.append((id_, num_err)) 
    

with open("./RL.csv", 'w') as csv_f: 
    csv_w = csv.writer(csv_f, delimiter = ',')   
    csv_w.writerows(id_score_list)

    
# with open(os.path.join(result_dir, csv_f), "r") as f:
# reader = csv.reader(f, delimiter=",")
        
    

    

    
    


    



    
            
        
    
    
