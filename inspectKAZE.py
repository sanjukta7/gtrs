ln = 5
df = []

import dataset
import local_binary_pattern
import util
import numpy as np
import greycomatrix
import greycoprops
import cv2
import gabor

subset1 = df.filter([
    'IMG_NAME', 'ORB', 'BRISK', 'KAZE', 'AKAZE', 'CONTRAST', 'DISSIMILARITY',
    'LBP_ENTROPY', 'HOMOGENEITY', 'LBP_ENERGY', 'GBR_ENTROPY', 'GBR_ENERGY',
    'ENERGY', 'hu', 'DETECT'
],
                    axis=1)

subset_train = subset_train.sort_values(by=['KAZE'])
subset_train['KAZE_L1'] = 0
subset1_list = np.vsplit(subset_train, ln)

subset_train['KAZE_L1'] = 0
k = 1
for row in range(len(subset_train)):
    i = 1
    for df1 in subset1_list:
        df1['KAZE_L1'] = i
        min1 = df1['KAZE'].min()
        max1 = df1['KAZE'].max()
        if k == 1:
            print("KAZE", df1['KAZE'].min(), df1['KAZE'].max())
        #print("KAZE", df1['KAZE'].min(), df1['KAZE'].max() )
        #print(df1.head())
        if (subset_train.loc[row, 'KAZE'] >=
                min1) & (subset_train.loc[row, 'KAZE'] < max1):
            subset_train.loc[row, 'KAZE_L1'] = i
        if (i == len(subset1_list)) & (subset_train.loc[row, 'KAZE'] >= min1):
            subset_train.loc[row, 'KAZE_L1'] = i
        i += 1
    k = 2