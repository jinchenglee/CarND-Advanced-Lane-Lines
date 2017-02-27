import numpy as np
import cv2

class lane()
    """
    Class to contain all lane features. 
    """

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

    def filter_luma(img, threshold = 30):
        """
        Return a img-sized binary file in which 1 represents 
        the pixel luminance greater than threshold.
        """
        assert img.ndim==2
        luma_binary = np.zeros_like(img)
        luma_binary[img>threshold]=1
        return luma_binary

    def filter_gradient_threshold(img, direction="x", threshold=(50,150),ksize=3)
        """
        Return a img-sized binary file in which 1 represents
        the gradient at specific pixel location is greater 
        than threshold. Taking "x" or "y" direction as input.
        """
        assert img.ndim==2
        # Sobel x on L channel
        if direction=="x":
            # Take the derivative in x dir
            sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize)
        else:
            # Take the derivative in y dir
            sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize)
        # Absolute x derivative to accentuate lines away from horizontal
        abs_sobel = np.absolute(sobel) 
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

        # Threshold x gradient
        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1


