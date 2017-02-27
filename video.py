import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
import sys
import img_filter 
import lane


# Lane detection pipeline
def pipeline(lane, img, fresh_start=False, luma_th=30, sat_th=(170, 255), grad_th=(50, 150), sobel_ksize=5):
    '''
    Processing pipeline
    
    1. Leverage HLS color space. 
    2. Gradients threshold.
    3. Area of interest mask.
    '''
    img = np.copy(img)

    # Get various parameters
    K,d,P,P_inv = lane.get_param()

    # Undistort
    img = img_filter.undistort(img, K, d)

    # Convert to HSV color space and separate the V channel
    hls = img_filter.conv_hls_halfmask(img)
    # Luma threshold
    luma_binary = np.zeros_like(hls[:,:,1])
    luma_binary = img_filter.filter_luma(hls[:,:,1], threshold=luma_th)

    # Sobel x on L channel
    grad_binary = np.zeros_like(luma_binary)
    grad_binary = img_filter.filter_gradient_threshold(image=hls[:,:,1],threshold=grad_th, ksize=sobel_ksize)

    # Threshold Saturation color channel
    sat_binary = np.zeros_like(luma_binary)
    sat_binary = img_filter.filter_sat(img_sat_ch=hls[:,:,2], threshold=sat_th)

    # Combine filter binaries
    binary = np.zeros_like(luma_binary)
    binary = img_filter.filter_fusion(luma_binary, sat_binary, grad_binary)

    # Perspective transform
    img_size = (binary.shape[1], binary.shape[0])
    binary_warped = cv2.warpPerspective(binary, P, img_size, flags=cv2.INTER_NEAREST)

    # Curve fit for the 1st frame
    if fresh_start:
        curve_fit_img = lane.curve_fit_1st(binary_warped)
        #print("left_fit = ", left_fit)
        #print("right_fit = ", right_fit)
    else:
        # Simulate the case to feed a "second" frame using curve_fit()
        curve_fit_img = lane.curve_fit(binary_warped)
        #print("left_fit = ", left_fit)
        #print("right_fit = ", right_fit)

    return curve_fit_img


# -------------------------------------
# Command line argument processing
# -------------------------------------
#if len(sys.argv) < 2:
#    print("Missing image file.")
#    print("python3 lane_detector.py <image_file>")
#
#FILE = str(sys.argv[1])
#
## Read in an test image
#image = mpimg.imread(FILE)


#clip = cv2.VideoCapture("project_video.mp4")
#clip = cv2.VideoCapture("frame_gt_900.avi")
clip = cv2.VideoCapture("frame_gt_500.avi")
#clip = cv2.VideoCapture("challenge_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

frame_cnt = 0
frame_start = 0
frame_end = 0xffffffff
#frame_end = 50

out=None
l_fit = 0
r_fit = 0

l_cnt = 0
r_cnt = 0

left_fit = []
right_fit = []

# Generate x and y values for plotting
ploty = np.linspace(0, 719, 720)

lane = lane.lane()

while True:
    flag, image = clip.read()
    if flag:
        frame_cnt += 1
        if frame_cnt < frame_start:
            continue
        elif frame_cnt > frame_end:
            break
        print('frame_cnt = ', frame_cnt)
        if out == None:
            out = cv2.VideoWriter('output.avi', fourcc, 30.0, (image.shape[1], image.shape[0]))

        # Video pipeline
        visualize_img = pipeline(lane, image, frame_cnt==1)
        #print("raw l_fit = ", l_fit, "raw r_fit = ", r_fit)

        ## Evaluate whether the new curvature values make sense.
        #NUM_HISTORY = 3
        #NUM_NOUPDATE = 6
        #if len(left_fit)<NUM_HISTORY:
        #    left_fit.append(l_fit)
        #else:
        #    l_avg = np.mean(np.array(left_fit), axis=0)
        #    l_std = np.std(np.array(left_fit), axis=0)
        #    l = np.abs((l_fit - l_avg)/l_avg) < 0.5
        #    if l.all():
        #        print("l updated.")
        #        left_fit.pop(0)
        #        left_fit.append(l_fit)
        #        l_cnt = 0
        #    else:
        #        l_fit = l_avg
        #        l_cnt += 1
        #        if l_cnt > NUM_NOUPDATE:
        #            print("l reset.")
        #            left_fit = []
        #            l_cnt = 0

        #if len(right_fit)<NUM_HISTORY:
        #    right_fit.append(r_fit)
        #else:
        #    r_avg = np.mean(np.array(right_fit), axis=0)
        #    r_std = np.std(np.array(right_fit), axis=0)
        #    r = np.abs((r_fit - r_avg)/r_avg) < 0.5
        #    if r.all():
        #        print("r updated.")
        #        right_fit.pop(0)
        #        right_fit.append(r_fit)
        #        r_cnt = 0
        #    else:
        #        r_fit = r_avg
        #        r_cnt += 1
        #        if r_cnt > NUM_NOUPDATE:
        #            print("r reset.")
        #            right_fit = []
        #            r_cnt = 0

        ## Bird's eye binary
        #color_binary = np.dstack((binary_warped, binary_warped, binary_warped))
        #res = np.array(color_binary*255, dtype='uint8')

        ##Visualize
        #margin = 80
        #nonzero = binary_warped.nonzero()
        #nonzeroy = np.array(nonzero[0])
        #nonzerox = np.array(nonzero[1])
        #left_lane_inds = ((nonzerox > (l_fit[0]*(nonzeroy**2) + l_fit[1]*nonzeroy + l_fit[2] - margin)) & (nonzerox < (l_fit[0]*(nonzeroy**2) + l_fit[1]*nonzeroy + l_fit[2] + margin))) 
        #right_lane_inds = ((nonzerox > (r_fit[0]*(nonzeroy**2) + r_fit[1]*nonzeroy + r_fit[2] - margin)) & (nonzerox < (r_fit[0]*(nonzeroy**2) + r_fit[1]*nonzeroy + r_fit[2] + margin)))  
        ## Generate x and y values for plotting
        #ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        #l_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
        #r_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]
        ## Create an image to draw on and an image to show the selection window
        #out_img = np.array(np.dstack((binary_warped, binary_warped, binary_warped))*255, dtype='uint8')
        #window_img = np.zeros_like(out_img)
        ## Color in left and right line pixels
        #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        ## Generate a polygon to illustrate the search window area
        ## And recast the x and y points into usable format for cv2.fillPoly()
        #left_line_window1 = np.array([np.transpose(np.vstack([l_fitx-margin, ploty]))])
        #left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([l_fitx+margin, ploty])))])
        #left_line_pts = np.hstack((left_line_window1, left_line_window2))
        #right_line_window1 = np.array([np.transpose(np.vstack([r_fitx-margin, ploty]))])
        #right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([r_fitx+margin, ploty])))])
        #right_line_pts = np.hstack((right_line_window1, right_line_window2))
        ## Draw the lane onto the warped blank image
        #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        #res1 = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        K,d,P,P_inv = lane.get_param()

        # Convert back to map to road
        l_fit = lane.current_fit[0]
        r_fit = lane.current_fit[1]
        left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
        right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]
        # Create an image to draw the lines on
        color_warp = np.zeros_like(visualize_img).astype(np.uint8)
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, P_inv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        # TODO: Need to alpha blending with undistorted image.
        res = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

        res = cv2.addWeighted(visualize_img, 1, res, 0.5, 0)

        # Write video out
        cv2.imshow('video', res)
        #print("l_fit = ", l_fit, "r_fit = ", r_fit)
        out.write(res)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

## File to save curvature fit data
#record = {}
#curvature_file = open('curvature.p','wb')
#record["l"] = left_fit
#record["r"] = right_fit
#pickle.dump(record, curvature_file)
#curvature_file.close()
#


