import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob

# ### Processing pipeline
# 
# Parameters to detect:
# 1. Leverage HLS color space. 
# 2. Gradients threshold.
# 3. Area of interest mask.

# Lane detection pipeline
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100), sobel_ksize=5):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    
    # AOI (area of interest) mask - we only care about lower part of the image
    size_x, size_y, size_ch = hsv.shape
    hsv[0:size_x//2,:,:] = 0
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]

    # Sobel x on L channel
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_ksize) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold Saturation color channel
    s_binary = np.zeros_like(s_channel)
    scaled_s_ch = np.uint8(255*s_channel/np.max(s_channel))
    s_binary[(scaled_s_ch >= s_thresh[0]) & (scaled_s_ch <= s_thresh[1])] = 1
    
    # Gradients direction 
    #dir_binary = dir_threshold(img, sobel_kernel=sobel_ksize, thresh=(np.pi/7, np.pi/3))

    # Stack each channel
    #color_binary = np.dstack((dir_binary, sxbinary, s_binary))
    #color_binary = np.dstack((np.zeros_like(dir_binary), np.zeros_like(sxbinary), s_binary))

    binary = np.zeros_like(s_channel)
    binary[(sxbinary==1) | (s_binary==1)] = 1
    return binary



# Read in an test image
image = mpimg.imread('test_images/test1.jpg')

# Undistort
f = open('camera_cal/wide_dist_pickle.p', 'rb')
param = pickle.load(f)
K = param["mtx"]        # Camera intrinsic matrix
d = param["dist"]       # Distortion parameters
image = cv2.undistort(image, K, d, None, K)
mpimg.imsave("test_images/test1_undistorted.png", image)

# Process lane detection filters
result = pipeline(image)

# Write binary image out
mpimg.imsave("test_images/test1_binary.png", result, cmap='gray')

