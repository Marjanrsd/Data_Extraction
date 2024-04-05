# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:08:51 2023

@author: Marjan
"""
# This code extracts perceived stress scores from a csv file
import os
import csv

with open("perceived4.csv", "r") as f:
    id_score_list = []
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        score = 0
        answered = 0
        id_ = 0
        for j, response in enumerate(line):
            if j == 0:
                if response == "ID":
                    break
                if response == "Participant number":
                    break
                id_ = response
                continue
            
            if j==4 or j==5 or j==7 or j==8:
                if response == "Never":
                    score+=4
                    answered += 1
                elif response == "Almost Never":
                    score += 3
                    answered += 1
                elif response == "Sometimes":
                    score += 2
                    answered += 1
                elif response == "Fairly Often":
                     score += 1
                     answered += 1
                elif response == "Very Often":
                     score += 0
                     answered += 1
                continue
            
            if response == "Never":
                score += 0
                answered += 1
            elif response == "Almost Never":
                score += 1
                answered += 1
            elif response == "Sometimes":
                score += 2
                answered += 1
            elif response == "Fairly Often":
                score += 3
                answered += 1
            elif response == "Very Often":
                 score += 4
                 answered += 1
           
        if answered == 0:
            continue
        perceived_score = score 
      
        id_score_list.append((id_, perceived_score))
    
 # will then save into csv wh/ each line is all MT neurons for a "trial"
with open("./perceived_scores.csv", 'w') as csv_f: 
    csv_w = csv.writer(csv_f, delimiter = ',')   
    # print(id_score_list)
    csv_w.writerows(id_score_list)
