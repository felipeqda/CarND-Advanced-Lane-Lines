# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:42:57 2019

@author: Felipe
"""


"""------------------------------------------------------------"""
"""imports"""
"""------------------------------------------------------------"""
#importing some useful packages
#set backend %matplotlib qt4 "ipython magic"
import matplotlib.pyplot as plt
plt.rcParams.update({'backend':'Qt4Agg'})
#modules
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from pdb import set_trace as stop #useful for debugging
import glob #to expand paths

"""------------------------------------------------------------"""
"""general use sub-routines"""
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
"""Step 1: Camera Calibration """
"""------------------------------------------------------------"""

def calibrateCamera(FORCE_REDO = False):
    """ Load images, get the corner positions in image and generate
        the calibration matrix and the distortion coefficients.
        if FORCE_REDO == False; reads previously saved .npz file, if available
    """
    #check if already done    
    if os.path.isfile('cal_para.npz') and (FORCE_REDO == False):
        #restore parameters if already done
        cal_para = np.load('cal_para.npz')        
        cal_mtx = cal_para['cal_mtx']
        dist_coef =  cal_para['dist_coef']
        cal_para.close()
    else:
        #find image names
        cal_images = glob.glob("camera_cal/*.jpg")
        chessb_corners = (9,6) #parameter for chessboard = (9,6) corners
        
        #define known chessboard points in normalized coordinates (3D): grid
        chessb_knownpoints = np.zeros([chessb_corners[0]*chessb_corners[1], 3],
                                     dtype=np.float32)
        chessb_knownpoints[:, 0:2] = np.mgrid[0:chessb_corners[0], 0:chessb_corners[1]].T.reshape(-1,2)
        
        #for each image, store known position and image position for calibration
        img_points_list = []
        known_points_list = []
        for img_path in cal_images:
            #load image
            image = mpimg.imread(img_path)
            Ny, Nx, _ = np.shape(image)     #image shape into number of lines, collumns
            #convert to grayscale
            grayscl = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #get chessboard positions
            ret, img_corners = cv2.findChessboardCorners(grayscl, chessb_corners, None)
            #add to list
            if ret: #points found
                known_points_list.append(chessb_knownpoints)
                img_points_list.append(img_corners)
        #for each image
        
        #apply calibration for a sample image (the last one loaded)
        #Get camera calibration parameters, given object points, image points, and the shape of the grayscale image:
        ret, cal_mtx, dist_coef, rvecs, tvecs = cv2.calibrateCamera(known_points_list,
                                                                    img_points_list,
                                                                    (Nx, Ny), None, None)
        #save parameters for posterior use
        np.savez('cal_para.npz', cal_mtx=cal_mtx, dist_coef=dist_coef, 
                 chessb_corners=chessb_corners)
        
        #Undistort a test image and save results:
        output_dir ='output_images'+os.sep+'calibration'+os.sep
        os.makedirs(output_dir, exist_ok = True)
            
        image = mpimg.imread('camera_cal\\calibration1.jpg')
        cal_image = cv2.undistort(image, cal_mtx, dist_coef, None, cal_mtx)

        #input image
        fig = plt.figure(num=1)
        fig.canvas.set_window_title('Input Image')
        plt.imshow(image)
        plt.savefig(output_dir+"cal_input.jpg", format='jpg')
        #calibrated image        
        fig = plt.figure(num=2)
        fig.canvas.set_window_title('Input Image after Calibration')
        plt.imshow(cal_image)
        plt.savefig(output_dir+"cal_output.jpg", format='jpg')

    #return calibration matrix and distortion coefficients to be used with
    #cv2.undistort(image, cal_mtx, dist_coef, None, cal_mtx)
    return cal_mtx, dist_coef
#------------------------------
    
cal_mtx, dist_coef = calibrateCamera(FORCE_REDO = False)



"""------------------------------------------------------------"""
"""Step 2: Single-frame pipeline """
"""------------------------------------------------------------"""

#image_list = os.listdir('./test_images')
image_list = ['straight_lines1.jpg',
              'straight_lines2.jpg',
              'test1.jpg',
              'test2.jpg',
              'test3.jpg',
              'test4.jpg',
              'test5.jpg',
              'test6.jpg']

input_image = "test_images"+os.sep + image_list[0]


SHOW_STEPS = True
sobel_kernel = 9

#take image name, make a directory for each, for output (used if plotting is on)
img_basename = os.path.basename(input_image).split('.jpg')[0]    
output_dir ='output_images'+os.sep+img_basename+os.sep
if SHOW_STEPS:
    os.makedirs(output_dir, exist_ok = True)

#read image and get dimensions
img_RGB = mpimg.imread(input_image)
Ny, Nx, _ = np.shape(img_RGB)

"""I) apply calibration"""
img_RGB = cv2.undistort(img_RGB, cal_mtx, dist_coef, None, cal_mtx)


"""II) color transformations & gradients """
"""II.I ==> verify channels"""
#compute grayscale and get basic gradients of grayscale
img_grayscl = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)
#get xx and yy gradients, scale amplitude to byte and threshold (using quizz sub-routines)
grad_x = abs_sobel_thresh(img_grayscl, orient='x', thresh=(20,100),
                          sobel_kernel = sobel_kernel, GRAY_INPUT=True)
grad_y = abs_sobel_thresh(img_grayscl, orient='y', thresh=(20,100),
                          sobel_kernel = sobel_kernel, GRAY_INPUT=True)
grad_xy = mag_thresh(img_grayscl, thresh=(20,100), GRAY_INPUT=True)
grad_dir = dir_thresh(img_grayscl, thresh=(0.7,1.3),
                      sobel_kernel = 19, GRAY_INPUT=True)

if SHOW_STEPS:
    #printing out some image info and plot
    plt.close('all')        
    print('Image file: ', img_basename, 'with dimensions:', img_RGB.shape)
    fig = plt.figure(num=0, figsize=(12,9))
    fig.canvas.set_window_title('Input image color/grayscale')
    plt.subplot(2,2,1)
    plt.imshow(img_grayscl, cmap = "gray") #cmap='gray' for single channel!
    plt.title('Grayscale')
    #show grayscale gradients in x and y
    plt.subplot(2,2,2)
    plt.imshow(grad_x, cmap = "gray") #cmap='gray' for single channel!
    plt.title('Grayscale - Grad_dx')
    plt.subplot(2,2,3)
    plt.imshow(grad_y, cmap = "gray") #cmap='gray' for single channel!
    plt.title('Grayscale - Grad_dy')
    plt.subplot(2,2,4)
    plt.imshow(grad_dir, cmap = "gray") #cmap='gray' for single channel!
    plt.title('Grayscale - dir')

    #plt.plot(x_ROI, y_ROI, 'r--', lw=2)


#check RGB channels
if SHOW_STEPS:
    #plot RGB channels for threshold analysis
    fig = plt.figure(num=2, figsize=(12,9))
    fig.canvas.set_window_title('Input image RGB color channels')
    plt.subplot(2,2,1)
    plt.imshow(img_RGB) 
    plt.title('RGB')
    plt.subplot(2,2,2)
    plt.imshow(img_RGB[:,:,0], cmap = "gray")  #cmap='gray' for single channel!
    plt.title('R')
    plt.subplot(2,2,3)
    plt.imshow(img_RGB[:,:,1], cmap = "gray")  #cmap='gray' for single channel!
    plt.title('G')
    plt.subplot(2,2,4)
    plt.imshow(img_RGB[:,:,2], cmap = "gray")  #cmap='gray' for single channel!
    plt.title('B')

#convert to HLS color space
img_HLS = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HLS)
#np.shape(img_HLS) = (Ny, Ny, 3 = [H, L, S])
if SHOW_STEPS:
    #plot HLS channels for threshold analysis
    fig = plt.figure(num=3, figsize=(12,9))
    fig.canvas.set_window_title('Input image HLS color channels')
    plt.subplot(2,2,1)
    plt.imshow(img_RGB) 
    plt.title('RGB')
    plt.subplot(2,2,2)
    plt.imshow(img_HLS[:,:,0], cmap = "gray")  #cmap='gray' for single channel!
    plt.title('H')
    plt.subplot(2,2,3)
    plt.imshow(img_HLS[:,:,1], cmap = "gray")  #cmap='gray' for single channel!
    plt.title('L')
    plt.subplot(2,2,4)
    plt.imshow(img_HLS[:,:,2], cmap = "gray")  #cmap='gray' for single channel!
    plt.title('S')

    #H/S + gradients of H and S
    fig = plt.figure(num=4, figsize=(12,9))
    fig.canvas.set_window_title('HS channels + gradients')
    plt.subplot(2,2,1)
    plt.imshow(img_HLS[:,:,0], cmap = "gray")  #cmap='gray' for single channel!
    plt.title('H')
    plt.subplot(2,2,2)
    plt.imshow(img_HLS[:,:,2], cmap = "gray")  #cmap='gray' for single channel!
    plt.title('S')
    plt.subplot(2,2,3)
    plt.imshow(mag_thresh(img_HLS[:,:,0], thresh=(10,100), GRAY_INPUT=True), cmap = "gray")  #cmap='gray' for single channel!
    plt.title('GradH')
    plt.subplot(2,2,4)
    plt.imshow(mag_thresh(img_HLS[:,:,2], thresh=(10,100), GRAY_INPUT=True), cmap = "gray") #cmap='gray' for single channel!
    plt.title('GradS')


"""II.II ==> apply thresholds and show masked result"""
#"positive" thresholds to include relevant marking
#high S value
S_thd = (200, 255)
S_mask = (img_HLS[:,:,2] > S_thd[0]) & (img_HLS[:,:,2] <= S_thd[1])
#high x-gradient
gradx_mask = abs_sobel_thresh(img_RGB, orient='x', thresh=(20,100), sobel_kernel = sobel_kernel)
#high R values
R_thd = (200, 255)
R_mask = (img_RGB[:,:,0] > R_thd[0]) & (img_RGB[:,:,0] <= R_thd[1])

mask = S_mask | gradx_mask | R_mask


#"negative" threshold to reject artifacts
roadside_mask = (img_HLS[:,:,0] < 40) & (img_HLS[:,:,2] > 30)  & (img_HLS[:,:,2] < 120) 



asphalt_mask = (img_HLS[:,:,0] > 100) & (img_HLS[:,:,2] < 30) 
#roadside_mask = (img_HLS[:,:,2] <= S_thd[0])


reject_mask = (~roadside_mask) & (~asphalt_mask)

if SHOW_STEPS:
    # Stack each channel to view their individual contributions in R/G/B
    color_binary = np.dstack(( R_mask, gradx_mask, S_mask)) * 255

    fig = plt.figure(num=5, figsize=(12,9))
    fig.canvas.set_window_title('Lane masks')
    plt.subplot(2,2,1)
    plt.imshow(color_binary) 
    plt.title('Masks')
    plt.subplot(2,2,2)
    plt.imshow(mask, cmap = "gray")  #cmap='gray' for single channel!
    plt.title('Output - add')
    plt.subplot(2,2,3)
    plt.imshow(reject_mask, cmap = "gray")  #cmap='gray' for single channel!
    plt.title('Filter')
    plt.subplot(2,2,4)
    plt.imshow(mask&reject_mask, cmap = "gray")  #cmap='gray' for single channel!
    plt.title('Output - filtered')
