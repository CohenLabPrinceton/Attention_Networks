import os
import csv

folder = '/tigress/jl40/save_june'
#folder = '/tigress/jl40/save_auto_blind_2'

subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
subfolders.sort()

good_folders = []
bad_folders = []
acc_all = []
acc_large = []
acc_top = []

for i in range(len(subfolders)):

    subdir = subfolders[i]

    subdir_split = subdir.split('/')
    dir_name = subdir_split[4]

    read_name = subdir + '/results_test.log'
    print(read_name)

    if os.path.isfile(read_name):
        # Add to good csv list: 
        good_folders.append(dir_name)

        with open(read_name) as fh:
            next(fh)
            parse_str = next(fh).split(' ')
            acc_all.append(parse_str[2])
            acc_large.append(parse_str[6][:-1])
            acc_top.append(parse_str[-1][:-2])
    else:
        # Add to bad csv list
        bad_folders.append(dir_name)

rows = zip(good_folders, acc_all, acc_large, acc_top)

with open('./acc_saver.csv', "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

with open('./acc_bad.csv', "w") as f:
    writer = csv.writer(f)
    for row in bad_folders:
        writer.writerow(row)



