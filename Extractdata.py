# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 12:09:50 2023

@author: Marjan (she's super cute!!)
"""

import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

og_data_mat = []


with open("PCA.csv", "r") as f:
    id_score_list = []
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        # ignore headers
        if i == 0:
            continue
        
        new_row = []
        for e in line:
            if e == '':
                continue
            new_row.append(float(e))
            
        if len(new_row) == 0:
            continue
       
        og_data_mat.append(new_row)
    
    data_mat = StandardScaler().fit_transform(og_data_mat)
    pca = PCA(n_components=2)
    new_data = pca.fit_transform(data_mat)
    
    for j in range(len(new_data)):
        new_x = new_data[j,0]
        new_y = new_data[j,1]
        num_err = og_data_mat[j][0]
     
        
        color = 'r' # low
        if num_err > 2:
            color = 'b' # medium
            if num_err > 10:
                color = 'g' # high
   # for color in ['red', 'blue', 'green']:
        plt.scatter(new_x, new_y, c=color)
        plt.title("2 Component PCA")
        plt.xlabel("Principle Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid()
        targets = ['Low', 'Medium', 'High']
        #colors=["red", "blue", "green"]
   # plt.legend(targets)
        
    
    print(pca.components_)
    
        
#         # print('line[{}] = {}'.format(i, line))
#         score = 0
#         answered = 0
#         id = 0
#         for j, response in enumerate(line):
#             if j == 1:
#                 if response == "test":
#                     break
#                 id = response
#                 continue
#             if response == "Not True":
#                 # score += 0
#                 answered += 1
#             elif response == "Somewhat True":
#                 score += 1
#                 answered += 1
#             elif response == "Very True":
#                 score += 2
#                 answered += 1
#             # else:
#                 # print(response, "\n")
#         if answered == 0:
#             continue
#         avg_score = score / answered
      
#         id_score_list.append((id, avg_score))
    
#  # will then save into csv wh/ each line is all MT neurons for a "trial"
# with open("./avg_scores.csv", 'w') as csv_f: 
#     csv_w = csv.writer(csv_f, delimiter = ',')   
#     # print(id_score_list)
#     csv_w.writerows(id_score_list)