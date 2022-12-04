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

subset_train = subset1.sort_values(by=['ORB'])
subset_train['ORB_L1'] = 0
print(subset_train)
print(len(subset_train))
subset1_list = np.vsplit(subset_train, ln)
print(len(subset1_list))
subset_train['ORB_L1'] = 0
k = 1
for row in range(len(subset_train)):
    i = 1
    for df1 in subset1_list:
        df1['ORB_L1'] = i
        min1 = df1['ORB'].min()
        max1 = df1['ORB'].max()
        if k == 1:
            print("ORB", df1['ORB'].min(), df1['ORB'].max())
        #print("ORB", df1['ORB'].min(), df1['ORB'].max())
        ##print(df1.head())
        if (subset_train.loc[row, 'ORB'] >=
                min1) & (subset_train.loc[row, 'ORB'] < max1):
            subset_train.loc[row, 'ORB_L1'] = i
        if (i == len(subset1_list)) & (subset_train.loc[row, 'ORB'] >= min1):
            subset_train.loc[row, 'ORB_L1'] = i
        i += 1
    k = 2