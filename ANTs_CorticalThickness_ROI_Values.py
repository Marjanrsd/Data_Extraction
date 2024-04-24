# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:49:20 2024

@author: marjan
"""
# This code extracts the cortical thickness values for each ROI for each subject from ANTs txt output
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

data = []
roi_values = {}
vals = []
vals2 = []
ids = []
result_file=r"D:\lizzzz\ants_CT.txt"
with open(result_file, "r") as f:
    ids = []
    id_value_list = []
    thick_rows = []
    reader = csv.reader(f, delimiter=",")
   
    for i, line in enumerate(reader):
        data.append(line)
        if any("Running -" in l for l in line):
            id_=line[0][9:13]
            ids.append(id_)
        elif any("Running " in l for l in line):
            id_=line[0][8:12]
            ids.append(id_)
         
        if id_ not in roi_values:
           roi_values[id_] = {"LH_Vis": [], "LH_SomMot":[],
                              "LH_DorsAttn_Post":[], "LH_DorsAttn_FEF":[], 
                              "LH_DorsAttn_PrCv":[],"LH_SalVentAttn_ParOper":[], 
                              "LH_SalVentAttn_PFCl":[],"LH_SalVentAttn_Med":[], 
                              "LH_Limbic_OFC":[], "LH_Limbic_TempPole":[],
                              "LH_Cont_Par":[], "LH_Cont_Temp":[], "LH_Cont_OFC":[],
                              "LH_Cont_PFCl":[], "LH_Cont_pCun":[], "LH_Cont_Cing":[],
                              "LH_Cont_PFCmp":[], "LH_Default_Temp":[], 
                              "LH_Default_Par":[], "LH_Default_PFC":[], 
                              "LH_Default_pCunPCC":[], "RH_Vis": [], "RH_SomMot":[],
                              "RH_DorsAttn_Post":[], "RH_DorsAttn_FEF":[], 
                               "RH_DorsAttn_PrCv":[], 
                               "RH_SalVentAttn_PFCl":[],"RH_SalVentAttn_Med":[], 
                               "RH_Limbic_OFC":[], "RH_Limbic_TempPole":[],
                               "RH_Cont_Par":[], "RH_Cont_Temp":[], 
                               "RH_Cont_PFCl":[], "RH_Cont_pCun":[], "RH_Cont_Cing":[],
                               "RH_Cont_PFCmp":[], "RH_Default_Temp":[], 
                               "RH_Default_Par":[], "RH_Default_PFC":[], 
                               "RH_Default_pCunPCC":[]}
            
        for key in roi_values[id_].keys():
            if any(key in l for l in line):
                print(key)
                line = next(reader, None)
                val = line
                val = val[0][:5]
                val = ''.join(char for char in val if char.isdigit() or char == '.')
                if val:  # Check if val is not an empty string
                    val = float(val)
                    roi_values[id_][key].append(val)
       
print(roi_values)      
averages = {}
for id_, rois in roi_values.items():
    averages[id_] = {}
    for roi, values in rois.items():
        avg = sum(values) / len(values)
        averages[id_][roi] = avg

print(averages)

# write it to a csv file
output_file = "new2_CT_ants.csv"   
rows = []
for id_, rois in averages.items():
    row = {"ID": id_}
    for roi, avg_value in rois.items():
        row[roi] = avg_value
    rows.append(row)

# Write the pivoted data to the CSV file
with open(output_file, "w", newline="") as csvfile:
    fieldnames = ["ID"] + list(next(iter(averages.values())).keys())  
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

            
