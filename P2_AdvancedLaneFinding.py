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
"""general use sub-rountines"""
"""------------------------------------------------------------"""


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
        ret, cal_mtx, dist_coef, rvecs, tvecs = cv2.calibrateCamera(known_points_list, img_points_list,
                                                               [Nx, Ny], None, None)
        #save parameters for posterior use
        np.savez('cal_para.npz', cal_mtx, dist_coef, chessb_corners)
        
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
    
cal_mtx, dist_coef = calibrateCamera()
