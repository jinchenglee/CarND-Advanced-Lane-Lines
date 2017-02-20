import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import pickle

def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped


# Read in an test image
image = mpimg.imread('test_images/straight_lines2.jpg')

# Undistort
f = open('camera_cal/wide_dist_pickle.p', 'rb')
param = pickle.load(f)
K = param["mtx"]        # Camera intrinsic matrix
d = param["dist"]       # Distortion parameters
image = cv2.undistort(image, K, d, None, K)

# Perspective transform 
img_size = (image.shape[1], image.shape[0])
# mapping points: src and dst in order of (top-left, bottom-left, bottom-right, top-right)
src = np.float32(
    [[(img_size[0] / 2) - 65, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
warped = warper(image, src, dst)

# Save warped image
mpimg.imsave('test_images/straight_lines2_warp.png', warped)

# Read in another test image to verify
image = mpimg.imread('test_images/straight_lines1.jpg')
warped = warper(image, src, dst)
mpimg.imsave('test_images/straight_lines1_warp.png', warped)
