# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 12:09:50 2023

@author: Marjan
"""
# This code extracts chronic stress scores from a csv file
import csv

with open("test4-chronic.csv", "r") as f:
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
                if response == "Participant":
                    break
                if response == "ID":
                    break
                id_ = response
                continue
            if response == "Not True":
                # score += 0
                answered += 1
            elif response == "Somewhat True":
                score += 1
                answered += 1
            elif response == "Very True":
                score += 2
                answered += 1
            # else:
                # print(response, "\n")
        if answered == 0:
            continue
        avg_score = score / answered
      
        id_score_list.append((id_, avg_score))
    
 # will then save into csv wh/ each line is all MT neurons for a "trial"
with open("./avg_scores.csv", 'w') as csv_f: 
    csv_w = csv.writer(csv_f, delimiter = ',')   
    # print(id_score_list)
    csv_w.writerows(id_score_list)
