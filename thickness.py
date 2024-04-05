import os
import csv

# This code extracts cortical thickness from the output of the FreeSurfer
data_dir = r"/mnt/chrastil/users/marjanrsd/corticalthickness/"
sub_dirs = os.listdir(data_dir)
sub_ids = []
hemispheres = ["rh", "lh"]
for d in sub_dirs:
    if len(d) == 4:
        try:
            int(d)
            # add if it can be interpreted as an int
            sub_ids.append(d)
        # if it can't, catch error and continue loop
        except:
            continue
# overwrite sub_dirs with filtered paths
sub_dirs = []
for i in sub_ids:
    sd = os.path.join(data_dir, i)
    sub_dirs.append(sd)

thick_rows = []
for sub_dir in sub_dirs:
    print("\nsub_dir: ", sub_dir)
    thick_row = []
    sub_id = sub_dir[-4:]
    thick_row.append(sub_id)
    for hem in hemispheres:
        print("\nhemisphere: ", hem)
        hem_stats = f'{hem}.aparc.stats'
        stats_file = os.path.join(sub_dir, 'stats/', hem_stats)

        with open(stats_file, "r+") as f:
            lines = f.readlines()
   
        roi_names = [
            "entorhinal",
            "parahippocampal",
            "inferiortemporal"
        ]
        for roi_name in roi_names:
            print("ROI: ", roi_name)
            for l in lines:
                if roi_name in l:
                    split_l = l.split(" ")
                    # filter blank/empty list elements
                    split_l = [x for x in split_l if x != '']
                    #print(split_l)
                    thick_avg = split_l[4]
                    thick_row.append(thick_avg)
                    print("avg thickness: ", thick_avg)
    thick_rows.append(thick_row)

filename = os.path.join(os.getcwd(), "thickness.csv")
col_header = []
for h in hemispheres:
    for r in roi_names:
        col_header.append(h + ' ' + r)
col_header = ["sub_ID"] + col_header
with open(filename, "w") as csv_f:
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(col_header)
    csv_writer.writerows(thick_rows)
