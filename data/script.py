import glob
import os

import numpy as np
import cv2


label_files = glob.glob('labels/*.png')

# labels are currently 0, 128, 255
# we want 0, 1, 2

# for label_file in label_files:
#     img = cv2.imread(label_file, 0)
#     img[img == 128] = 1
#     img[img == 255] = 2
#     cv2.imwrite(label_file, img)


# # print the unique values in the labels
# unique_labels = []
# for label_file in label_files:
#     img = cv2.imread(label_file, 0)
#     print(np.unique(img))


import matplotlib.pyplot as plt

img = cv2.imread('data/labels/0.png')
plt.imshow(img*50)

plt.show()