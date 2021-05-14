# coding: utf-8

import numpy as np
import cv2
import random

"""
    Cnum images were randomly selected to calculate mean and standard deviation STD by channel
    First normalize the pixels from 0 to 255 to 0-1 to calculate
"""


train_csv_path = '../all_train_img.csv'

CNum = 2000     # How much picture is selected for calculation

img_h, img_w = 256, 256
imgs = np.zeros([img_w, img_h, 1, 1])
means, stdevs = [], []

with open(train_csv_path, 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)   # shuffle , Random selection picture

    for i in range(CNum):
        img_path = lines[i].rstrip().split(',')[1]

        img = cv2.imread(img_path)
        img = img[:,:,0]
        print(img.shape)
        img = cv2.resize(img, (img_h, img_w))

        img = img[:, :, np.newaxis, np.newaxis]
        print(img.shape)
        imgs = np.concatenate((imgs, img), axis=3)
        print(i)

imgs = imgs.astype(np.float32)/255.


pixels = imgs[:,:,0,:].ravel()  # Line up
means.append(np.mean(pixels))
stdevs.append(np.std(pixels))

# means.reverse() # BGR --> RGB
# stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
