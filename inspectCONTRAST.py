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

subset_train = subset_train.sort_values(by=['CONTRAST'])
subset_train['CONTRAST_L1'] = 0
subset1_list = np.vsplit(subset_train, ln)

subset_train['CONTRAST_L1'] = 0
for row in range(len(subset_train)):
    i = 1
    for df1 in subset1_list:
        df1['CONTRAST_L1'] = i
        min1 = df1['CONTRAST'].min()
        max1 = df1['CONTRAST'].max()
        #print("KAZE", df1['KAZE'].min(), df1['KAZE'].max() )
        #print(df1.head())
        if (subset_train.loc[row, 'CONTRAST'] >=
                min1) & (subset_train.loc[row, 'CONTRAST'] < max1):
            subset_train.loc[row, 'CONTRAST_L1'] = i
        i += 1