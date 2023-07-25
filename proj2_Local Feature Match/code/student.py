import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
import cv2
import math

def ANMS (x , y, r, maximum):
    i = 0
    j = 0
    NewList = []
    while i < len(x):
        minimum = 1000000000000
        # 获得一个harris角点的横纵坐标，称为基础点
        X, Y = x[i], y[i]
        while j < len(x):  # 遍历除该角点外每一个得分（R值）更高的角点，称为比较点，找到离基础点最近的一个比较点，记录下基础点的横纵坐标和与最近比较点之间的距离
            CX, CY = x[j], y[j]
            if (X != CX and Y != CY) and r[i] < r[j]:
                distance = math.sqrt((CX - X)**2 + (CY - Y)**2)
                if distance < minimum:
                    minimum = distance
            j = j + 1
        NewList.append([X, Y, minimum])
        i = i + 1
        j = 0
    # 根据距离大小对基础点进行排序，很显然，距离越小的点说明在该角点周围有更好的角点，所以可以适当舍弃。在舍弃一定数目的得分较小的角点后，就得到了非最大抑制后的harris角点坐标。
    NewList.sort(key = lambda t: t[2])
    NewList = NewList[len(NewList)-maximum:len(NewList)]
    return NewList

def get_interest_points(image, feature_width):
    '''
    Returns a set of interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here!
    XCorners = []
    YCorners = []
    RValues = []
    alpha = 0.04
    threshold = 10000
    ImageRows = image.shape[0]
    ImageColumns = image.shape[1]
    imX = cv2.Sobel(image, cv2.CV_64F,1,0,ksize=5)
    imY = cv2.Sobel(image, cv2.CV_64F,0,1,ksize=5)
    Ixx = (imX) * (imX)
    Iyy = (imY) * (imY)
    Ixy = (imX) * (imY)
    for i in range(16, ImageRows - 16):
        for j in range(16, ImageColumns - 16):
            Ixx1 = Ixx[i-1:i+1, j-1:j+1]
            Iyy1 = Iyy[i-1:i+1, j-1:j+1]
            Ixy1 = Ixy[i-1:i+1, j-1:j+1]
            Ixxsum = Ixx1.sum()
            Iyysum = Iyy1.sum()
            Ixysum = Ixy1.sum()
            Determinant = Ixxsum*Iyysum - Ixysum**2	#计算矩阵的特征值
            Trace = Ixxsum + Iyysum	#计算矩阵的迹
            R = Determinant - alpha*(Trace**2) #角点相应函数R
            #判断每个像素相应函数是否大于阈值R，如果大于阈值则合格。
            if R > threshold:
                XCorners.append(j)
                YCorners.append(i)
                RValues.append(R)
    XCorners = np.asarray(XCorners)
    YCorners = np.asarray(YCorners)
    RValues = np.asarray(RValues)
    NewCorners = ANMS(XCorners, YCorners, RValues, 3025)
    NewCorners = np.asarray(NewCorners)


    # These are placeholders - replace with the coordinates of your interest points!
    xs = NewCorners[:,0]
    ys = NewCorners[:,1]
    return xs,ys


def get_features(image, x, y, feature_width):
    '''
    Returns a set of feature descriptors for a given set of interest points.

    (Please note that we reccomend implementing this function after you have implemented
    match_features)

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each descriptor_window_image_width/4.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''

    # TODO: Your implementation here! 
    x = np.rint(x)
    x = x.astype(int)
    y = np.rint(y)
    y = y.astype(int)
    filter1 = cv2.getGaussianKernel(ksize=4,sigma=10)
    filter1 = np.dot(filter1, filter1.T)
    image = cv2.filter2D(image, -1, filter1)
    ImageRows = image.shape[0]
    ImageColumns = image.shape[1]
    xlen = len(x)
    ylen = len(y)
    FeatureVectorIn = np.ones((xlen,128))
    NormalizedFeature = np.zeros((ylen,128))
    for i in range(xlen):
        temp1 = int(x[i])
        temp2 = int(y[i])
        Window = image[temp2-8:temp2 + 8, temp1-8:temp1 + 8]
        WindowRows = Window.shape[0]
        WindowColumns = Window.shape[1]
        for p in range(4):
            for q in range(4):
                WindowCut = Window[p*4:p*4 +4,q*4: q*4+4]
                NewWindowCut = cv2.copyMakeBorder(WindowCut, 1, 1, 1, 1, cv2.BORDER_REFLECT)
                Magnitude = np.zeros((4,4))
                Orientation = np.zeros((4,4))
                for r in range(WindowCut.shape[0]):
                    for s in range(WindowCut.shape[1]):
                        Magnitude[r,s] = math.sqrt((NewWindowCut[r+1,s] - NewWindowCut[r-1,s])**2 + (NewWindowCut[r,s+1] - NewWindowCut[r,s-1])**2)
                        Orientation[r,s] = np.arctan2((NewWindowCut[r+1,s] - NewWindowCut[r-1,s]),(NewWindowCut[r,s+1] - NewWindowCut[r,s-1]))
                Magnitude = Magnitude
                OrientationNew = Orientation*(180/(math.pi))
                hist, edges = np.histogram(OrientationNew, bins = 8, range = (-180,180), weights = Magnitude)
                for t in range(8):
                    l = t+p*32+q*8
                    FeatureVectorIn[i,l] = hist[t]
    for a in range(FeatureVectorIn.shape[0]):
        sum1 = 0
        for b in range(FeatureVectorIn.shape[1]):
            sum1 = sum1 + (FeatureVectorIn[a][b])*(FeatureVectorIn[a][b])
        sum1 = math.sqrt(sum1)
        for c in range(FeatureVectorIn.shape[1]):
            NormalizedFeature[a][c] = FeatureVectorIn[a][c]/sum1

    # This is a placeholder - replace this with your features!
    features = NormalizedFeature
    return features


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - zip (python built in function)

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here!
    Distance = np.zeros((im1_features.shape[0], im2_features.shape[0]))
    Value = []
    Hitx = []
    Hity = []
    for x in range(im1_features.shape[0]):
        for y in range(im2_features.shape[0]):
            ExtractedRow1 = im1_features[[x],:]
            ExtractedRow2 = im2_features[[y],:]
            SubtractedRow = ExtractedRow1 - ExtractedRow2
            Square = SubtractedRow*SubtractedRow
            Sum = Square.sum()
            Sum = math.sqrt(Sum)
            Distance[x,y] = Sum
        IndexPosition = np.argsort(Distance[x,:])
        d1 = IndexPosition[0]
        d2 = IndexPosition[1]
        Position1 = Distance[x,d1]
        Position2 = Distance[x,d2]
        ratio = Position1/Position2
        if ratio<0.8: 
            Hitx.append(x)
            Hity.append(d1)
            Value.append(Position1)
    Xposition = np.asarray(Hitx)
    Yposition = np.asarray(Hity)
    matches = np.stack((Xposition,Yposition), axis = -1)
    confidences = np.asarray(Value)
    return matches, confidences

    # These are placeholders - replace with your matches and confidences!
