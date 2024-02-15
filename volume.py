import os
import csv

data_dir = r"/mnt/chrastil/users/marjanrsd/corticalthickness/"
sub_dirs= os.listdir(data_dir)
#print(sub_dirs)
sub_ids = []
hemispheres = ["rh", "lh"]
for d in sub_dirs:
    if len(d) == 4:
        try:
            int(d)
            sub_ids.append(d)
        except:
            continue
#print(sub_ids)

# overwrite sub_dirs with new paths 
sub_dirs = []
for i in sub_ids:
   new_path = os.path.join(data_dir, i)
   sub_dirs.append(new_path)

vol_rows = [] #volume rows
for sub_dir in sub_dirs:
    vol_row = []
    sub_id = sub_dir[-4:]
    vol_row.append(sub_id)
    print(vol_row)
    sub_stats = "aseg.stats"
    stats_file = os.path.join(sub_dir, "stats/", sub_stats)
    print(stats_file)
    with open(stats_file, "r+") as f:
         lines = f.readlines()

    roi_names = [
    "Left-Caudate",
    "Left-Putamen",
    "Left-Hippocampus",
    "Left-Amygdala",
    "Right-Caudate",
    "Right-Putamen",
    "Right-Hippocampus",
    "Right-Amygdala"
    ]
    for roi_name in roi_names:
        print("ROI :", roi_name)
        for l in lines:
            if roi_name in l:
                split_l = l.split(" ")
                split_l = [x for x in split_l if x !=""]
                volume = split_l[3]
                vol_row.append(volume)
                print("Volume(mm3):", volume)
    vol_rows.append(vol_row)

filename = os.path.join(os.getcwd(),"volume.csv")
col_header = []
for r in roi_names:
    col_header.append(r)
col_header = ["sub_ID"] + col_header
with open(filename, "w") as csv_f:
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(col_header)
    csv_writer.writerows(vol_rows)

           



    


    








