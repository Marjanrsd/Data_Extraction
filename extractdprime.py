# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 08:06:25 2023

@author: Marjan
"""

import os
import csv

results_dir = r"C:\Users\Marjan\Documents\liz\Data analysis\n2 spatial"
all_CSVs = os.listdir(results_dir)
id_dprime_list = []
for csv_f in all_CSVs:
    id_ = csv_f[:4]
    print(id_)
    with open(os.path.join(results_dir, csv_f), "r") as f:
        reader = csv.reader(f, delimiter=",")
        response_col = 0
        score = 0
        answered = 0
        for i, line in enumerate(reader):
            
            if i == 0:
                for j, response in enumerate(line):
                    if response == "rating.response":
                        response_col = j
                        break
                continue
            
            for j, response in enumerate(line):
                
                if j != response_col:
                    continue
                if response == "":
                    continue
                
                score = score + int(response)
                answered = answered + 1
            # end for loop
        # end for loop
                
        avg_score = score / answered
        id_score_list.append((id_, avg_score))
    

with open("./anxiety_avg_scores.csv", 'w') as csv_f: 
    csv_w = csv.writer(csv_f, delimiter = ',')   
    csv_w.writerows(id_score_list)