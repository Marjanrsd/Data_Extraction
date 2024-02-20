#THIS IS A CODE FOR CALCULATING HIPPOCAMPAL VOLUME FROM CSV FILES
import csv
import os

hem =["L", "R"]
total_volume_dict = {}
for i in range (1063, 1064):
    for j in hem:
        out_dir = os.path.join(os.getcwd(), "VolumeCalc", f"{i}_{j}.csv")
        print(out_dir)
        with open(out_dir, "r") as f:
            csv_reader = csv.reader(f)
            vol = 0
            
            for row_num, row in enumerate(csv_reader):
                if row_num in [2, 3, 4, 8, 9]:
                    cell_vol = float(row[3])
                    vol+= cell_vol
            sub_key = f"{i}_{j}"
            total_volume_dict[sub_key] = vol
            

for subject, volume in total_volume_dict.items():
    print(f"Subject {subject}: Total Volume {volume}")

with open("itksnapvol.csv", "w", newline = "")  as file:
    writer = csv.writer(file)
    #write the header row
    writer.writerow(["Subject", "Volume"])
    for subject, volume in total_volume_dict.items():
        writer.writerow([subject, volume])

