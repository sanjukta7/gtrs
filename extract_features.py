df = []
import dataset
import local_binary_pattern
import util
import numpy as np
import greycomatrix
import greycoprops
import cv2
import gabor

for img in dataset:
    i = i + 1
    features = []
    img_arr = img[0]
    img_arr = util.img_as_ubyte(img_arr)
    print(img_arr)
    feat_lbp = local_binary_pattern(img_arr, 8, 1, 'uniform')
    feat_lbp = np.uint8((feat_lbp / feat_lbp.max()) * 255)
    lbp_hist, _ = np.histogram(feat_lbp, 8)
    lbp_hist = np.array(lbp_hist, dtype=float)
    lbp_prob = np.divide(lbp_hist, np.sum(lbp_hist))
    lbp_energy = np.sum(lbp_prob**2)
    lbp_entropy = -np.sum(np.multiply(lbp_prob, np.log2(lbp_prob)))
    distance = [1, 2, 3, 4]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    gCoMat = greycomatrix(np.array(img_arr),
                          distances=distance,
                          angles=angles,
                          symmetric=True,
                          normed=True)
    contrast = greycoprops(gCoMat, prop='contrast')
    dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
    homogeneity = greycoprops(gCoMat, prop='homogeneity')
    energy = greycoprops(gCoMat, prop='energy')
    correlation = greycoprops(gCoMat, prop='correlation')
    moment = cv2.moments(np.array(img_arr))
    hu = cv2.HuMoments(moment)
    gaborFilt_real, gaborFilt_imag = gabor(img_arr, frequency=0.6)
    gaborFilt = (gaborFilt_real**2 + gaborFilt_imag**2) // 2
    gabor_hist, _ = np.histogram(gaborFilt, 8)
    gabor_hist = np.array(gabor_hist, dtype=float)
    gabor_prob = np.divide(gabor_hist, np.sum(gabor_hist))
    gabor_energy = np.sum(gabor_prob**2)
    gabor_entropy = -np.sum(np.multiply(gabor_prob, np.log2(gabor_prob)))
    features.append(img[4])
    features.append("{:.5f}".format(lbp_energy))
    features.append("{:.5f}".format(lbp_entropy))
    features.append("{:.5f}".format(contrast[0][0]))
    features.append("{:.5f}".format(dissimilarity[0][0]))
    features.append("{:.5f}".format(homogeneity[0][0]))
    features.append("{:.5f}".format(energy[0][0]))
    features.append("{:.5f}".format(correlation[0][0]))
    features.append("{:.5f}".format(gabor_energy))
    features.append("{:.5f}".format(gabor_entropy))
    features.append(img[1])
    features.append(img[2])
    features.append(img[3])
    features.append(img[6])
    features.append(img[5])
    df.append(features)
