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
        visualize_img = lane.curve_fit_1st(binary_warped)
        #print("left_fit = ", left_fit)
        #print("right_fit = ", right_fit)
    else:
        # Simulate the case to feed a "second" frame using curve_fit()
        visualize_img = lane.curve_fit(binary_warped)
        #print("left_fit = ", left_fit)
        #print("right_fit = ", right_fit)

    # Draw detected lane onto the road
    res = lane_shadow_on_road_img = lane.draw_lane_area(binary_warped, img, P_inv)

    # Optional: blending with visualization image
    res = cv2.addWeighted(res, 1, visualize_img, 0.5, 0)

    return res


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
        res = pipeline(lane, image, frame_cnt==1)
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


