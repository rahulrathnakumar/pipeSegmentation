import os
import random
import shutil

import glob
# set working directory
os.chdir('data/')


rgb_files = glob.glob('rgb/*.png')
depth_files = glob.glob('depth/*.png')
mean_curvature_files = glob.glob('mean_curvature/*.png')
gaussian_curvature_files = glob.glob('gaussian_curvature/*.png')
labels_files = glob.glob('labels/*.png')
normal_files = glob.glob('normal/*.png')

# assert all files are the same length
assert len(rgb_files) == len(depth_files) == len(mean_curvature_files) == len(gaussian_curvature_files) == len(labels_files) == len(normal_files)


rgb_files = sorted(rgb_files)
depth_files = sorted(depth_files)
mean_curvature_files = sorted(mean_curvature_files)
gaussian_curvature_files = sorted(gaussian_curvature_files)
labels_files = sorted(labels_files)
normal_files = sorted(normal_files)

assert ([os.path.basename(rgb_file)[:15] for rgb_file in rgb_files]) == ([os.path.basename(depth_file)[:15] for depth_file in depth_files]) == ([os.path.basename(mean_curvature_file)[:15] for mean_curvature_file in mean_curvature_files]) == ([os.path.basename(gaussian_curvature_file)[:15] for gaussian_curvature_file in gaussian_curvature_files]) == ([os.path.basename(labels_file)[:15] for labels_file in labels_files]) == ([os.path.basename(normal_file)[:15] for normal_file in normal_files])


# replace the files in all folders with the number in the loop

for i in range(len(rgb_files)):
    # rename the files
    os.rename(rgb_files[i], f'rgb/{i}.png')
    os.rename(depth_files[i], f'depth/{i}.png')
    os.rename(mean_curvature_files[i], f'mean_curvature/{i}.png')
    os.rename(gaussian_curvature_files[i], f'gaussian_curvature/{i}.png')
    os.rename(labels_files[i], f'labels/{i}.png')
    os.rename(normal_files[i], f'normal/{i}.png')

    