# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:13:28 2019

@author: Felipe
"""

"""Collect sub-routines used as tools"""


""" Imports """
#modules
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from pdb import set_trace as stop #useful for debugging
import glob #to expand paths


"""------------------------------------------------------------"""
""" Image display                                              """
"""------------------------------------------------------------"""
#---------------------------------------------------------------------
def weighted_img(img_annotation, initial_img, α=0.8, β=1., γ=0.):
###mixes two color images of the same shape, img and initial_img, for annotations
    return cv2.addWeighted(initial_img, α, img_annotation, β, γ)
#---------------------------------------------------------------------
def gaussian_blur(img, kernel_size):
    ###Applies a Gaussian Noise kernel###
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
#---------------------------------------------------------------------
def restrict2ROI(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, np.int32([vertices]), ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
#---------------------------------------------------------------------

def plotNimg(imgs, titles, cmap, fig_title, fig_num=None):
    #plot a stack of images with given titles and color maps
    #cmap='gray' for single channel!
    N_img = len(imgs)
    N_row = int(np.ceil(N_img/2.0))
    fig = plt.figure(num=fig_num, figsize=(12,4.5*N_row))
    fig.canvas.set_window_title(fig_title)
    
    for i_img in range(N_img):
        plt.subplot(N_row,2,i_img+1)
        plt.imshow(imgs[i_img], cmap = cmap[i_img]) 
        plt.title(titles[i_img])

    #TODO: save!
#---------------------------------------------------------------------
    
"""------------------------------------------------------------"""
""" Masking and Thresholding Image or Gradient                 """
"""------------------------------------------------------------"""

#Threshold application, as developed in quizzes
#---------------------------------------------------------------------
def abs_sobel_thresh(img, orient='y', thresh = (0,255),
                     GRAY_INPUT = False, sobel_kernel=3):
# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
#---------------------------------------------------------------------
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    if GRAY_INPUT == False:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
       
    
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    #use cf2.CV_64F (double) to avoid problem with negative (white2black) derivatives!
    if orient in ['x', 'y']: #x or y direction
        direction = {'x': (1,0), 'y': (0,1)}
        sobel_grad = cv2.Sobel(img_gray, cv2.CV_64F,
                               direction[orient][0], direction[orient][1],
                               ksize = 2*(sobel_kernel//2)+1)
    else:
        raise Exception('Unknown direction: '+orient)
    # 3) Take the absolute value of the derivative or gradient
    sobel_grad = np.abs(sobel_grad)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel_byte = np.uint8(255*sobel_grad/np.max(sobel_grad))

    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    if thresh == (None, None):
        mask = sobel_byte #return byte image (to check levels)
    else:        
        mask = (sobel_byte > thresh[0]) & (sobel_byte < thresh[1])
    # 6) Return this mask as your binary_output image
    return mask
#---------------------------------------------------------------------

#---------------------------------------------------------------------
def mag_thresh(img, sobel_kernel=3, thresh=(0, 255), GRAY_INPUT=False):
# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
#---------------------------------------------------------------------    
    # Apply the following steps to img
    # 1) Convert to grayscale
    if GRAY_INPUT == False:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    # 2) Take the gradient in x and y separately
    gradx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=2*(sobel_kernel//2)+1) 
    grady = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=2*(sobel_kernel//2)+1)
    # 3) Calculate the magnitude 
    grad_mag = np.sqrt(gradx**2 + grady**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    grad_byte = np.uint8(255*grad_mag/np.max(grad_mag))
    # 5) Create a binary mask where mag thresholds are met
    mask = (grad_byte > thresh[0]) & (grad_byte < thresh[1])
    # 6) Return this mask as your binary_output image
    return mask
#---------------------------------------------------------------------

#---------------------------------------------------------------------
def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2), 
               GRAY_INPUT = False):
# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
#---------------------------------------------------------------------    
    # Apply the following steps to img
    # 1) Convert to grayscale
    if GRAY_INPUT == False:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    # 2) Take the gradient in x and y separately
    gradx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize =2*(sobel_kernel//2)+1)
    grady = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize =2*(sobel_kernel//2)+1)
    # 3) Take the absolute value of the x and y gradients
    grad_absx = np.abs(gradx)
    grad_absy = np.abs(grady)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_dir = np.arctan2(grad_absy, grad_absx)

    # 5) Create a binary mask where direction thresholds are met    
    if thresh == (None, None):
        mask = grad_dir #return byte image (to check levels)
    else:        
        mask = (grad_dir > thresh[0]) & (grad_dir < thresh[1])
    # 6) Return this mask as your binary_output image
    return mask
#---------------------------------------------------------------------


"""------------------------------------------------------------"""
""" Miscellaneous                                              """
"""------------------------------------------------------------"""
def closePolygon(pts_xy):
    #get a poligon with [N, 2] vertices and returns x and y for plotting
    #by repeating the last point
    x = np.concatenate((pts_xy[:,0], [pts_xy[0,0]]) )
    y = np.concatenate((pts_xy[:,1], [pts_xy[0,1]]) )
    return x, y