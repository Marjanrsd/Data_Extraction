# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:58:36 2023

@author: Marjan
"""
# This code extracts d prime score for n-back verbal task from csv files
import os
import csv
from scipy import stats
from scipy.stats import norm

results_dir = r"D:\lizzzz\n back verbal2"
all_CSVs = os.listdir(results_dir)
d_prime_score = []
for csv_f in all_CSVs:
    id_ = csv_f[:4]
    print(id_)
    with open(os.path.join(results_dir, csv_f), "r") as f:
        reader = csv.reader(f, delimiter=",")
        response_col = 0
        correct_ans_col=0
        correct_1s = 0
        total_1s = 0 # i.e. actual 1s from correct col
        incorrect_1s=0
        total_0s=0
        
        for i, line in enumerate(reader):
            # parse the header row
            if i == 0:
                for j, response in enumerate(line):
                    if response == "resp_test.keys":
                        response_col = j
                        break
                for k, response2 in enumerate(line):
                    if response2=="Ans":
                       correct_ans_col= k
                       break
                continue
            
            # non-header rows, i.e. the interesting data :)
            for j, response in enumerate(line):
                if j != response_col:
                    continue
                # once here, we know we're looking at the response column
                if response == "":
                    continue
                # once here, response cell is non-empty
                # now compare two columns (correct & participant's answer)
                if line[correct_ans_col] == "1":
                    total_1s = total_1s + 1
                    if  line[response_col] == line[correct_ans_col]:
                        correct_1s = correct_1s + 1
                if line[correct_ans_col] == "0":
                    total_0s = total_0s + 1
                    if  line[response_col] =="1":
                        incorrect_1s=incorrect_1s + 1
        
        hit_rate = correct_1s / total_1s
        print("hi")
        # don't know why but not supposed to have perfect hit rate :shrug:
        if hit_rate == 1:
            n = 40 # hard-coded, part of task design
            hit_rate = 1 - (1 / (2*n))
        print("hit_rate: ", hit_rate)

        false_alarm_rate = incorrect_1s/total_0s
        if false_alarm_rate==0:
           
            n = 40
            false_alarm_rate=(1/(2*n))
        print("false_alarm_rate: ", false_alarm_rate)
        d_prime = norm.ppf(hit_rate)-norm.ppf(false_alarm_rate)  
        print("d_prime:", d_prime)
        d_prime_score.append((id_,d_prime))
                
with open("./d_primee_verbal.csv", 'w') as csv_f: 
    csv_w = csv.writer(csv_f, delimiter = ',')   
    csv_w.writerows(d_prime_score)          
        
