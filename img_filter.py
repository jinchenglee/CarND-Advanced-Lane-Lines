import numpy as np
import cv2

class img_filter():
    """
    Class to host all image operations.
    """

    def __init__(self):
        pass
        
    def conv_hls_halfmask(self,image):
        """
        Convert input image to HLS colorspace and mask off upper half.
        Return converted image in three channels: H(0), L(1), S(2).
        """
        image = np.copy(image)
        # Convert to HSV color space and separate the V channel
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
        # AOI (area of interest) mask - we only care about lower part of the image
        size_x, size_y, size_ch = hls.shape
        hls[0:size_x//2,:,:] = 0
        return hls

    def filter_luma(self, image_luma, threshold = 30):
        """
        Return a image-sized binary file in which 1 represents 
        the pixel luminance greater than threshold.
        """
        assert image_luma.ndim==2
        luma_binary = np.zeros_like(image_luma)
        luma_binary[image_luma>threshold]=1
        return luma_binary

    def filter_sat(self, img_sat_ch, threshold = (170,255)):
        """
        Return a image-sized binary file in which 1 represents 
        the saturation at the pixel location within threshold.
        Expect a S channel input from HLS colorspace converted image.
        """
        assert img_sat_ch.ndim==2
        sat_binary = np.zeros_like(img_sat_ch)
        scaled_s_ch = np.uint8(255*img_sat_ch/np.max(img_sat_ch))
        sat_binary[(scaled_s_ch >= threshold[0]) & (scaled_s_ch <= threshold[1])] = 1
        return sat_binary    
    
    def filter_gradient_threshold(self, image, direction="x", threshold=(50,150),ksize=3):
        """
        Return a image-sized binary file in which 1 represents
        the gradient at specific pixel location is greater 
        than threshold. Taking "x" or "y" direction as input.
        """
        assert image.ndim==2
        # Sobel x on L channel
        if direction=="x":
            # Take the derivative in x dir
            sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize)
        else:
            # Take the derivative in y dir
            sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize)
        # Absolute x derivative to accentuate lines away from horizontal
        abs_sobel = np.absolute(sobel) 
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Threshold x gradient
        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1
        return sobel_binary

    def filter_fusion(self, luma_bin, sat_bin, grad_bin):
        """
        Fuse binary filters result.
        """
        binary = np.zeros_like(luma_bin)
        binary[((grad_bin==1) | (sat_bin==1)) & (luma_bin==1)] = 1
        return binary

