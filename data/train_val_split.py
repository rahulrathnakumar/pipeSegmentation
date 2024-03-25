import os
import random
import shutil

import glob
# set working directory
os.chdir('data/')

train_val_split = 0.65

# create a txt file with a list of files in train dataset

rgb_files = glob.glob('rgb/*.png')

# shuffle the files
seed = 42
random.seed(seed)
random.shuffle(rgb_files)

# rgb_files basename
rgb_files = [os.path.basename(file) for file in rgb_files]

# split the files
train_files = rgb_files[:int(len(rgb_files)*train_val_split)]
val_files = rgb_files[int(len(rgb_files)*train_val_split):]

# assert intersection is empty
assert len(set(train_files).intersection(set(val_files))) == 0


# write the files to a txt file
with open('train.txt', 'w') as f:
    for file in train_files:
        f.write(file + '\n')
        
with open('val.txt', 'w') as f:
    for file in val_files:
        f.write(file + '\n')