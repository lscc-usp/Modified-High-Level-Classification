import glob
import cv2
import numpy as np
import pandas as pd
from sklearn import preprocessing
from quadtree import QuadTreeSolution
import matplotlib.pyplot as plt # only for debug

silent = False  # Display track of progress info (when False)
# Paths
pathNormal = r"Input\NORMAL\*.png"             # Normal Chest X-ray Images
pathCOVID = r"Input\COVID-19\*.png"            # COVID-19 Chest X-ray Images
export_location_data = r"Output\data.csv"
export_location_data_target = r"Output\data_target.csv"

# Threshold for Fractal Dimension
min_value, max_value, steps = 100, 160, 10
Threshold = [x for x in range(min_value, max_value, steps)]


def convertjpg(pngfile, class_num, data, data_target, width=1024, height=1024, mode = 'hybrid'):
    
    assert mode == 'fractal' or mode == 'quadtree' or mode == 'hybrid'

    img = cv2.imread(pngfile, cv2.IMREAD_GRAYSCALE) # Importing as GrayScale
    #cv2.imshow('image', img)   # for debug
    #cv2.waitKey(0)
    x = []

    if mode=='fractal' or mode=='hybrid':
        imgFract = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)   # Bilinear interpolation method
        for i in Threshold:
            ret, b_img = cv2.threshold(imgFract, i, 255, cv2.THRESH_BINARY)
            d = fractal_dimension(b_img, i)
            x.append(d)
        x = np.array(x)

    if mode=='quadtree' or mode == 'hybrid':
        imgQuad = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
        s = QuadTreeSolution(imgQuad, 1, 60)
        his = s.extract_img_features() # Processed histogram
        #img_grid = s.bfs_for_segmentation() # Processed image (for DEBUG or Analysis)
        x = np.append(x,his)

    data.append(x)
    data_target.append(class_num)


def get_dimension(img):
    x = []
    for i in Threshold:
        y = fractal_dimension(img, i)
        x.append(y)
    return x


def fractal_dimension(Z, threshold):
    # Only for 2D images
    assert(len(Z.shape) == 2) # check 2D

    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0), # row
                               np.arange(0, Z.shape[1], k), axis=1)  # column

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    Z = (Z < threshold) # Transform Z into a binary array
    #plt.imshow(Z,interpolation='nearest',cmap=plt.cm.binary) # for debug

    p = min(Z.shape)                        # Minimal dimension of image
    n = 2**np.floor(np.log(p)/np.log(2))    # Greatest power of 2 less than or equal to p
    n = int(np.log(n)/np.log(2))            # Extracting the exponent
    sizes = 2**np.arange(n, 1, -1)          # Build successive box sizes (from 2**n down to 2**1)

    counts = [] # Actual box counting, with decreasing size
    for size in sizes:
        counts.append(boxcount(Z, size))

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1) # Fit the successive log(sizes) with log (counts)
    return -coeffs[0]


def get_data(mode = 'hybrid', is_preprocessed = False):

    def read_data(path, sampleSize, classID, className = '', mode = 'hybrid'):
        className = str(classID) if className=='' else className
        data= []
        data_target = []
        count = 0
        for pngfile in glob.glob(path):
            convertjpg(pngfile, classID, data, data_target, mode = mode)
            count += 1
            if not silent: print('Converting '+className+' class image '+str(count)+' of '+str(sampleSize)+'.\r', end="")
            if count == sampleSize:
                break
        if not silent: print()
        return data, data_target

    if is_preprocessed:
        df_data = pd.read_csv(export_location_data, sep=',', header=None)
        df_data_target = pd.read_csv(export_location_data_target, sep=',', header=None)

        return df_data.to_numpy(), df_data_target.to_numpy().ravel()

    dataNormal, data_targetNormal = read_data(pathNormal, 150, 0, 'Normal', mode)
    dataCOVID, data_targetCOVID = read_data(pathCOVID, 150, 1, 'COVID-19', mode)

    data = np.vstack(( np.array(dataNormal), np.array(dataCOVID) ))
    data_target = np.array(data_targetNormal + data_targetCOVID) # Simple Vector
    
    save = pd.DataFrame(data)
    save.to_csv(export_location_data, index=False, header=False)
    save = pd.DataFrame(data_target)
    save.to_csv(export_location_data_target, index=False, header=False)

    return data, data_target


def data_preprocess(data):
    """Feature engineering: Normalize"""
    scaler = preprocessing.MinMaxScaler().fit(data)
    data = scaler.transform(data)
    return data