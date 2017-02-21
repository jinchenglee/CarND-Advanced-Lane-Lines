##Advance Lane Finding

---

As part of Udacity Autonomous Driving Nano-Degree, the goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./camera_cal/calibration1.jpg "Distorted"
[image1]: ./camera_cal/test_undist.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Original"
[image2a]: ./test_images/test1_undistorted.png "Road Transformed"
[image3]: ./test_images/test1_binary.png "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image4a]: ./test_images/straight_lines2_warp.png "Warp Example 2"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image5a]: ./test_images/test1_curve_fit_window_search.png "Searching windows"
[image5b]: ./examples/histogram_lane_pixels.png "Lane pixels histogram"
[image6]: ./examples/example_output.jpg "Output"
[image7]: ./camera_cal/added_rgb_axis.jpg
[video1]: ./project_video.mp4 "Video"

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The OpenCV checkerboard calibration program is based on Professor Z. Zhang's paper "Z. Zhang. "A flexible new technique for camera calibration".". 

OpenCV implementation source code can be found [here](https://github.com/opencv/opencv/blob/master/modules/calib3d/src/calibration.cpp). 

The code for this step is contained in lines 18 through 71 of file camera_cal.py.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result.

Original image:
![alt text][image0]
 
Undistorted image:
![alt text][image1]

Draw the "world coordinates axises" on one of the checkerboard image.

```python
	...
	# OpenCV API returns:
	#  mtx (intrinsic matrix), dist (distortion parameters), 
	#  rvecs (Rodriguez vector), tvecs (Translation vector)
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
	...
	
	# --------------------
	# World co-ordinates of axis points
	# --------------------
	objp = np.array([
    	[0.,0.,0.], # Origin
    	[3.,0.,0.], # X (red axis)
    	[0.,3.,0.], # Y (green axis)
    	[0.,0.,-3.] # -Z (blue axis)
	])

	# Project above points to a specific image
	image_pt, _ = cv2.projectPoints(objp, rvecs[0], tvecs[0], mtx, dist)

```

Axises are lines connecting above points. Please be noticed the Z axis is negative. Also, the abosolute dimension is missing here, which is unnecessary to derive camera intrinsic and extrinsic matrices in calibration process for mono-camera using Zhang's method. 

![alt text][image7]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
Loading previously saved parameters in pickle file when I did camera calibration using checkerboard, I used OpenCV API undistort to do image distortion-correction before any further processing as below. 

```python
# Undistort
f = open('camera_cal/wide_dist_pickle.p', 'rb')
param = pickle.load(f)
K = param["mtx"]        # Camera intrinsic matrix
d = param["dist"]       # Distortion parameters
image = cv2.undistort(image, K, d, None, K)
mpimg.imsave("test_images/test1_undistorted.png", image)
```

The distortion-corrected image example:
![alt text][image2a]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of AOI (area of interest), color and gradient thresholds to generate a binary image.  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

```python
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

    # Combine the output - OR the two selections.
    binary = np.zeros_like(s_channel)
    binary[(sxbinary==1) | (s_binary==1)] = 1
```
![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in top lines in the file `perspective.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
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
...
    # Get the perspective change matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Apply the perspective change
    warped_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 705, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

Verified on another sample image after warping:
![alt text][image4a]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this in lane_detector.py:

![alt text][image5]

It starts with histogram of image columns. the left/right lane pixels should have histogram peaks in left and right side if searching from mid-point. 
```python
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

```
The peaks are set to be starting base points of left/right lanes. 
![alt text][image5b]

Then a window searching approach is taken to gradually search upwards from the left/right base point to find points clustered right above. These newly found points are meaned to find new base points for next round of window search upwards. There are a total of 9 layers of windows as below. 

![alt text][image5a]

All the lane line pixels are saved for polynominal fit calculation.
```python
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
```

The `left_fit/right_fit` values (shape=[1,3]) are updated for each input frame image. Except for the very 1st frame when left_fit/right_fit don't exist, all follow up frames can do "smarter" window search along previous frame fit curve.

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

