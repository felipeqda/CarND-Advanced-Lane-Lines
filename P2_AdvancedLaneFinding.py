# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:42:57 2019

@author: Felipe
"""

"""------------------------------------------------------------"""
"""imports"""
"""------------------------------------------------------------"""
# importing some useful packages

# set backend %matplotlib qt4 "ipython magic"
import matplotlib.pyplot as plt

plt.rcParams.update({'backend': 'Qt5Agg'})
# modules
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from pdb import set_trace as stop  # useful for debugging
import glob  # to expand paths

# threshold application
from P2_subroutines import abs_sobel_thresh, mag_thresh, dir_thresh
# image display
from P2_subroutines import plotNimg, weighted_img, restrict2ROI
from P2_subroutines import closePolygon
from P2_subroutines import find_lane_xy_frompoly, find_lane_xy_frommask

"""------------------------------------------------------------"""
"""general use sub-routines"""
"""------------------------------------------------------------"""

"""------------------------------------------------------------"""
"""Step 1: Camera Calibration """
"""------------------------------------------------------------"""


def calibrateCamera(FORCE_REDO=False):
    """ Load images, get the corner positions in image and generate
        the calibration matrix and the distortion coefficients.
        if FORCE_REDO == False; reads previously saved .npz file, if available
    """
    # check if already done
    if os.path.isfile('cal_para.npz') and (FORCE_REDO == False):
        # restore parameters if already done
        cal_para = np.load('cal_para.npz')
        cal_mtx = cal_para['cal_mtx']
        dist_coef = cal_para['dist_coef']
        cal_para.close()
    else:
        # find image names
        cal_images = glob.glob("camera_cal/*.jpg")
        chessb_corners = (9, 6)  # parameter for chessboard = (9,6) corners

        # define known chessboard points in normalized coordinates (3D): grid
        chessb_knownpoints = np.zeros([chessb_corners[0] * chessb_corners[1], 3],
                                      dtype=np.float32)
        chessb_knownpoints[:, 0:2] = np.mgrid[0:chessb_corners[0], 0:chessb_corners[1]].T.reshape(-1, 2)

        # for each image, store known position and image position for calibration
        img_points_list = []
        known_points_list = []
        for img_path in cal_images:
            # load image
            image = mpimg.imread(img_path)
            Ny, Nx, _ = np.shape(image)  # image shape into number of lines, collumns
            # convert to grayscale
            grayscl = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # get chessboard positions
            ret, img_corners = cv2.findChessboardCorners(grayscl, chessb_corners, None)
            # add to list
            if ret:  # points found
                known_points_list.append(chessb_knownpoints)
                img_points_list.append(img_corners)
        # for each image

        # apply calibration for a sample image (the last one loaded)
        # Get camera calibration parameters, given object points, image points, and the shape of the grayscale image:
        ret, cal_mtx, dist_coef, rvecs, tvecs = cv2.calibrateCamera(known_points_list,
                                                                    img_points_list,
                                                                    (Nx, Ny), None, None)
        # save parameters for posterior use
        np.savez('cal_para.npz', cal_mtx=cal_mtx, dist_coef=dist_coef,
                 chessb_corners=chessb_corners)

        # Undistort a test image and save results:
        output_dir = 'output_images' + os.sep + 'calibration' + os.sep
        os.makedirs(output_dir, exist_ok=True)

        image = mpimg.imread('camera_cal\\calibration1.jpg')
        cal_image = cv2.undistort(image, cal_mtx, dist_coef, None, cal_mtx)

        # input image
        fig = plt.figure(num=1)
        fig.canvas.set_window_title('Input Image')
        plt.imshow(image)
        plt.savefig(output_dir + "cal_input.jpg", format='jpg')
        # calibrated image
        fig = plt.figure(num=2)
        fig.canvas.set_window_title('Input Image after Calibration')
        plt.imshow(cal_image)
        plt.savefig(output_dir + "cal_output.jpg", format='jpg')

    # return calibration matrix and distortion coefficients to be used with
    # cv2.undistort(image, cal_mtx, dist_coef, None, cal_mtx)
    return cal_mtx, dist_coef


# ------------------------------

cal_mtx, dist_coef = calibrateCamera(FORCE_REDO=False)

"""------------------------------------------------------------"""
"""Step 2: Single-frame pipeline """
"""------------------------------------------------------------"""

# image_list = os.listdir('./test_images')
image_list = ['straight_lines1.jpg',
              'straight_lines2.jpg',
              'test1.jpg',
              'test2.jpg',
              'test3.jpg',
              'test4.jpg',
              'test5.jpg',
              'test6.jpg']

input_image = "test_images" + os.sep + image_list[5] #5!  #indices [0-7]

SHOW_STEPS = True
sobel_kernel = 7

# take image name, make a directory for each, for output (used if plotting is on)
img_basename = os.path.basename(input_image).split('.jpg')[0]
output_dir = 'output_images' + os.sep + img_basename + os.sep
if SHOW_STEPS:
    os.makedirs(output_dir, exist_ok=True)

# read image and get dimensions
img_RGB = mpimg.imread(input_image)
Ny, Nx, _ = np.shape(img_RGB)

"""I) apply calibration"""
img_RGB = cv2.undistort(img_RGB, cal_mtx, dist_coef, None, cal_mtx)

"""II) color transformations & gradients """
"""II.I ==> verify channels"""
# compute grayscale and get basic gradients of grayscale
img_grayscl = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)
# get xx and yy gradients, scale amplitude to byte and threshold (using quizz sub-routines)
grad_x = abs_sobel_thresh(img_grayscl, orient='x', thresh=(20, 100),
                          sobel_kernel=sobel_kernel, GRAY_INPUT=True)
grad_y = abs_sobel_thresh(img_grayscl, orient='y', thresh=(20, 100),
                          sobel_kernel=sobel_kernel, GRAY_INPUT=True)
grad_xy = mag_thresh(img_grayscl, thresh=(20, 100), GRAY_INPUT=True)
grad_dir = dir_thresh(img_grayscl, thresh=(0.7, 1.3),
                      sobel_kernel=19, GRAY_INPUT=True)

if SHOW_STEPS:
    # printing out some image info and plot
    plt.close('all')
    print('Image file: ', img_basename, 'with dimensions:', img_RGB.shape)

    plotNimg([img_grayscl, grad_x, grad_y, grad_dir],
             ['Grayscale', 'Grad_x', 'Grad_y', 'Grad_dir'],
             ["gray", "gray", "gray", "gray"],
             'Input image color/grayscale', fig_num=0)
    plt.savefig(output_dir + img_basename + '_grads.png')

# check RGB channels
if SHOW_STEPS:
    # plot RGB channels for threshold analysis
    plotNimg([img_RGB, img_RGB[:, :, 0], img_RGB[:, :, 1], img_RGB[:, :, 2]],
             ['RGB', 'R', 'G', 'B'],
             [None, "gray", "gray", "gray"],
             'Input image RGB color channels', fig_num=2)
    plt.savefig(output_dir + img_basename + '_RGB.png')

# convert to HLS color space
img_HLS = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HLS)
# output ==> np.shape(img_HLS) = (Ny, Ny, 3 = [H, L, S])

if SHOW_STEPS:
    # plot HLS channels for threshold analysis
    plotNimg([img_RGB, img_HLS[:, :, 0], img_HLS[:, :, 1], img_HLS[:, :, 2]],
             ['RGB', 'H', 'L', 'S'],
             [None, "gray", "gray", "gray"],
             'Input image HLS color channels', fig_num=3)
    plt.savefig(output_dir + img_basename + '_HLS.png')

    # H/S + gradients of H and S
    plotNimg([img_HLS[:, :, 0], img_HLS[:, :, 2],
              mag_thresh(img_HLS[:, :, 0], thresh=(10, 100), GRAY_INPUT=True),
              mag_thresh(img_HLS[:, :, 2], thresh=(10, 100), GRAY_INPUT=True)],
             ['H', 'S', 'Grad-H', 'Grad-S'],
             ["gray", "gray", "gray", "gray"],
             'HS channels + gradients', fig_num=4)

"""II.II ==> apply thresholds and show masked result"""
# "positive" thresholds to include relevant marking
# high S value
S_thd = (200, 255)
S_mask = (img_HLS[:, :, 2] > S_thd[0]) & (img_HLS[:, :, 2] <= S_thd[1])
# high x-gradient
gradx_mask = abs_sobel_thresh(img_RGB, orient='x', thresh=(20, 100), sobel_kernel=sobel_kernel)

# high S x-gradient
Sx_mask = abs_sobel_thresh(img_HLS[:, :, 2], orient='x', thresh=(20, 100), sobel_kernel=sobel_kernel, GRAY_INPUT=True)


# high R values ==> not robust against "bright" asphalt!
R_thd = (200, 255)
R_mask = (img_RGB[:, :, 0] > R_thd[0]) & (img_RGB[:, :, 0] <= R_thd[1])

mask = S_mask | gradx_mask | Sx_mask #| R_mask
#mask = S_mask | Sx_mask #| R_mask


##"negative" threshold to reject artifacts?
# roadside_mask = (img_HLS[:,:,0] < 40) & (img_HLS[:,:,2] > 30)  & (img_HLS[:,:,2] < 120)
# asphalt_mask = (img_HLS[:,:,0] > 100) & (img_HLS[:,:,2] < 30)
##roadside_mask = (img_HLS[:,:,2] <= S_thd[0])
# reject_mask = (~roadside_mask) & (~asphalt_mask)


"""III) Perspective Transformations and Region of Interest (ROI) """

# compute a perspective transform M, given source and destination points:

# based on a Ny x Nx = 720 x 1280 image (straight_lines.jpg)
pts_img = np.float32([[190 + 1, 720], [600 + 1, 445],
                      [680 - 2, 445], [1120 - 2, 720]])

# make closed polygons for plotting
x_img, y_img = closePolygon(pts_img)

pts_warp = np.float32([[350, 720], [350, 0], [950, 0], [950, 720]])
# make a closed polygon for plotting
x_warp, y_warp = closePolygon(pts_warp)

# direct perspective transform
M = cv2.getPerspectiveTransform(pts_img, pts_warp)
# inverse perspective transform:
Minv = cv2.getPerspectiveTransform(pts_warp, pts_img)

# Warp an image using the perspective transform, M:
img_warped = cv2.warpPerspective(img_RGB, M, (Nx, Ny), flags=cv2.INTER_LINEAR)

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 600  # meters per pixel in x dimension

# define ROI for spatial filtering using inverse transform
# cf. equations at
# https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#getperspectivetransform
# auxiliar 3D coordinates are used with z =1 for source and a normalization factor t for the destination
# transposition is done to facilitate matrix product
# take the warp region as a reference and expand, to get a rectangle in the top-down view
xmin, xmax = pts_warp[[0,2],0]
ymin, ymax = pts_warp[[2,3],1]
pts_warpedROI = np.float32([[xmin-200, ymax, 1], [xmin-200, ymin, 1], [xmax+200, ymin, 1], [xmax+200, ymax, 1]]).T
pts_ROI = np.tensordot(Minv.T, pts_warpedROI, axes=([0], [0]))
pts_ROI = (pts_ROI[0:2, :] / pts_ROI[2, :]).T
x_wROI, y_wROI = closePolygon(pts_warpedROI[0:2, :].T)
x_ROI, y_ROI = closePolygon(pts_ROI)

# TODO: save ROI, M, etc

if SHOW_STEPS:
    fig = plt.figure(figsize=(12, 4.5), num=5)
    fig.canvas.set_window_title('Perspective Transform')
    plt.subplot(1, 2, 1)
    plt.imshow(img_RGB)
    plt.plot(x_img, y_img, 'r--')
    plt.plot(x_ROI, y_ROI, 'b')
    plt.title('Input Image')
    plt.subplot(1, 2, 2)
    plt.imshow(img_warped)
    plt.plot(x_warp, y_warp, 'r--')
    plt.plot(x_wROI, y_wROI, 'b')
    plt.title('Warped "top-down" Image')
    plt.savefig(output_dir + img_basename + '_warp.png')

"""IV) Get restricted and warped detection masks"""

mask_img = restrict2ROI(img_RGB, pts_ROI)
mask_ROI = restrict2ROI(255 * np.uint8(mask), pts_ROI)
mask_warp = cv2.warpPerspective(mask_ROI, M, (Nx, Ny), flags=cv2.INTER_LINEAR)

if SHOW_STEPS:
    # Stack each channel to view their individual contributions in R/G/B
    color_binary = np.dstack((Sx_mask, gradx_mask, S_mask)) * 255

    plotNimg([color_binary, mask, mask_ROI, mask_warp],
             ['Masks', 'Output (+ masks)', 'ROI-restricted', 'Warped'],
             [None, "gray", "gray", "gray"],
             'Lane masks', fig_num=6)


# warp before detect ==> less blurry
S_warp = cv2.warpPerspective(img_HLS[:,:,2], M, (Nx, Ny), flags=cv2.INTER_LINEAR)
# high S value
S_thd = (200, 255)
S_mask2 = (S_warp > S_thd[0]) & (S_warp <= S_thd[1])
# high S x-gradient
Sx_mask2 = abs_sobel_thresh(S_warp, orient='x', thresh=(20, 100), sobel_kernel=sobel_kernel, GRAY_INPUT=True)
# high x-gradient
gradx_mask2 = abs_sobel_thresh(img_warped, orient='x', thresh=(20, 100), sobel_kernel=sobel_kernel)

color_binary_warp = np.dstack((Sx_mask2, gradx_mask2, S_mask2)) * 255
mask_warp2 = Sx_mask2 | gradx_mask2 | S_mask2


plt.close('all')

fig = plt.figure(figsize=(12, 4.5), num=7)
fig.canvas.set_window_title('Warp before detect?')
plt.subplot(1, 2, 1)
plt.imshow(color_binary_warp)
plt.subplot(1, 2, 2)
plt.imshow(mask_warp2, cmap = 'gray')



# treatment of mask
mask_in = 255*np.uint8(mask_warp2)

kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(mask_in,kernel,iterations = 1)

# remove noise by applying Morphological open
mask_mostreliable = cv2.morphologyEx(mask_in, cv2.MORPH_OPEN, kernel, iterations = 2)

points_SCORE = np.zeros([Ny, Nx], dtype=np.uint8)
points_SCORE +=  mask_in//255 #score 1 to points in mask
points_SCORE +=  10 * (cv2.morphologyEx(mask_in, cv2.MORPH_OPEN, kernel, iterations = 1)//255)
points_SCORE +=  20 * (cv2.morphologyEx(mask_in, cv2.MORPH_OPEN, kernel, iterations = 2)//255)



plt.figure(num=0)
left, right, out_img = find_lane_xy_frommask(mask_mostreliable)
print('Left fit: ', left.cf)
print('Right fit: ', right.cf)
plt.imshow(out_img)

plt.figure(num=1)
left, right, out_img = find_lane_xy_frommask(255*np.uint8(gradx_mask2))
print('Left fit: ', left.cf)
print('Right fit: ', right.cf)
plt.imshow(out_img)


#find_lane_xy_frompoly(mask_in, cf1, cf2)