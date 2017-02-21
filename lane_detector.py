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

def curve_fit_1st(binary_warped):
    '''
    Curve fit for 1st frame, using searching windows.
    '''
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Histogram analysis
    #plt.plot(histogram)
    #plt.show()
    
    # Create an output image to draw on and  visualize the result
    out_img = np.array(np.dstack((binary_warped, binary_warped, binary_warped))*255, dtype='uint8')
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Visualization
    ## Generate x and y values for plotting
    #ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    #left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    #
    #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    #plt.savefig('test_images/test1_curve_fit_window_search.png')

    return out_img, left_fit, right_fit

def curve_fit(binary_warped, left_fit, right_fit):
    '''
    Curve fit since 2nd frame
    '''
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    ##Visualize
    ## Generate x and y values for plotting
    #ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    #left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    ## Create an image to draw on and an image to show the selection window
    #out_img = np.array(np.dstack((binary_warped, binary_warped, binary_warped))*255, dtype='uint8')
    #window_img = np.zeros_like(out_img)
    ## Color in left and right line pixels
    #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    ## Generate a polygon to illustrate the search window area
    ## And recast the x and y points into usable format for cv2.fillPoly()
    #left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    #left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    #left_line_pts = np.hstack((left_line_window1, left_line_window2))
    #right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    #right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    #right_line_pts = np.hstack((right_line_window1, right_line_window2))
    ## Draw the lane onto the warped blank image
    #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    #result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    #plt.imshow(result)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)    
    #plt.savefig("test_images/test1_curve_fit_2.png")

    return left_fit, right_fit

# Read in an test image
image = mpimg.imread('test_images/test1.jpg')

# Undistort
f = open('camera_cal/wide_dist_pickle.p', 'rb')
param = pickle.load(f)
K = param["mtx"]        # Camera intrinsic matrix
d = param["dist"]       # Distortion parameters
f.close()
image = cv2.undistort(image, K, d, None, K)
mpimg.imsave("test_images/test1_undistorted.png", image)

# Process lane detection filters
image_binary = pipeline(image)

# Write binary image out
mpimg.imsave("test_images/test1_binary.png", image_binary, cmap='gray')

# Perspective transform
warp_f = open('camera_cal/warp.p', 'rb')
warp_param = pickle.load(warp_f)
P = warp_param["warp"]
warp_f.close()

img_size = (image_binary.shape[1], image_binary.shape[0])
binary_warped = cv2.warpPerspective(image_binary, P, img_size, flags=cv2.INTER_NEAREST)
mpimg.imsave("test_images/test1_binary_warp.png", binary_warped, cmap='gray')

# Curve fit for the 1st frame
curve_fit_img, cur_left_fit, cur_right_fit = curve_fit_1st(binary_warped)
#mpimg.imsave("test_images/test1_curve_fit_1.png", curve_fit_img)
print("left_fit = ", cur_left_fit)
print("right_fit = ", cur_right_fit)

# Simulate the case to feed a "second" frame using curve_fit()
left_fit, right_fit = curve_fit(binary_warped, cur_left_fit, cur_right_fit)
print("left_fit = ", left_fit)
print("right_fit = ", right_fit)

