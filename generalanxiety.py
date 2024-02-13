# -*- coding: utf-8 -*-
"""
Created on Tue May 16 01:42:12 2023

@author: Marjan
"""
import os
import csv
with open("general_anxiety4.csv", "r") as f:
    id_score_list = []
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        # print('line[{}] = {}'.format(i, line))
        score = 0
        answered = 0
        id_ = 0
        for j, response in enumerate(line):
            if j == 0:
                if response == "test":
                    break
                if response == "ID":
                    break
                if response == "Participant number":
                    break
                id_ = response
                continue
            
            if j==1 or j==3 or j==6 or j==7 or j==10 or j==13 or j==14 or j==16 or j==19:
                if response == "Almost never":
                    score+=4
                    answered += 1
                elif response == "Sometimes":
                    score += 3
                    answered += 1
                elif response == "Often":
                    score += 2
                    answered += 1
                elif response == "Almost always":
                     score += 1
                     answered += 1
                continue
            
            if response == "Almost never":
                score += 1
                answered += 1
            elif response == "Sometimes":
                score += 2
                answered += 1
            elif response == "Often":
                score += 3
                answered += 1
            elif response == "Almost always":
                 score += 4
                 answered += 1
            # else:
                # print(response, "\n")
        if answered == 0:
            continue
        anx_score = score 
      
        id_score_list.append((id_, anx_score))
    
 # will then save into csv wh/ each line is all MT neurons for a "trial"
with open("./GeneralAnxiety_scores.csv", 'w') as csv_f: 
    csv_w = csv.writer(csv_f, delimiter = ',')   
    # print(id_score_list)
    csv_w.writerows(id_score_list)