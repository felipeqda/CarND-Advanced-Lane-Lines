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

## INPUTIMAGE
input_image = "test_images" + os.sep + image_list[2] #5!  #indices [0-7]


# keywords to control level of output (set True for step by step analysis of image/frame)
SHOW_COLOR_GRADIENT = False  # show color channel analysis and gradients
SHOW_WARP = False            # show warping
SHOW_FIT = True              # show pixel detection and fitting step
sobel_kernel = 7

# take image name, make a directory for each, for output (used if plotting is on)
img_basename = os.path.basename(input_image).split('.jpg')[0]
output_dir = 'output_images' + os.sep + img_basename + os.sep
if SHOW_COLOR_GRADIENT:
    os.makedirs(output_dir, exist_ok=True)

# read image and get dimensions
img_RGB = mpimg.imread(input_image)
Ny, Nx, _ = np.shape(img_RGB)
# print image info and plot
plt.close('all')
print('')
print('Input Image file: ', img_basename, 'with dimensions:', img_RGB.shape)


"""I) apply calibration"""
img_RGB = cv2.undistort(img_RGB, cal_mtx, dist_coef, None, cal_mtx)


"""II) color transformations & gradients """
"""II.I ==> verify channels (optional)"""
# plot detailed analysis (useful to verify properties and select thresholds)
if SHOW_COLOR_GRADIENT:

    # compute grayscale image and get basic gradients of it
    img_grayscl = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)
    # get xx and yy gradients, scale amplitude to byte and threshold (using quiz sub-routines)
    grad_x = abs_sobel_thresh(img_grayscl, orient='x', thresh=(20, 100),
                              sobel_kernel=sobel_kernel, GRAY_INPUT=True)
    grad_y = abs_sobel_thresh(img_grayscl, orient='y', thresh=(20, 100),
                              sobel_kernel=sobel_kernel, GRAY_INPUT=True)
    grad_xy = mag_thresh(img_grayscl, thresh=(20, 100), GRAY_INPUT=True)
    grad_dir = dir_thresh(img_grayscl, thresh=(0.7, 1.3),
                          sobel_kernel=19, GRAY_INPUT=True)

    # plot grayscale images and gradients
    plotNimg([img_grayscl, grad_x, grad_y, grad_dir],
             ['Grayscale', 'Grad_x', 'Grad_y', 'Grad_dir'],
             ["gray", "gray", "gray", "gray"],
             'Input image color/grayscale', fig_num=0)
    plt.savefig(output_dir + img_basename + '_grads.png')

    # check RGB channels
    # plot RGB channels for threshold analysis
    plotNimg([img_RGB, img_RGB[:, :, 0], img_RGB[:, :, 1], img_RGB[:, :, 2]],
             ['RGB', 'R', 'G', 'B'],
             [None, "gray", "gray", "gray"],
             'Input image RGB color channels', fig_num=2)
    plt.savefig(output_dir + img_basename + '_RGB.png')

    # convert to HLS color space
    img_HLS = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HLS)
    # output ==> np.shape(img_HLS) = (Ny, Ny, 3 = [H, L, S])

    # plot HLS channels for threshold analysis
    plotNimg([img_RGB, img_HLS[:, :, 0], img_HLS[:, :, 1], img_HLS[:, :, 2]],
             ['RGB', 'H', 'L', 'S'],
             [None, "gray", "gray", "gray"],
             'Input image HLS color channels', fig_num=3)
    plt.savefig(output_dir + img_basename + '_HLS.png')

    # plot H/S + gradients of H and S
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

    # high R values ==> not used, not robust against "bright" asphalt!
    R_thd = (200, 255)
    R_mask = (img_RGB[:, :, 0] > R_thd[0]) & (img_RGB[:, :, 0] <= R_thd[1])

    mask = S_mask | gradx_mask | Sx_mask #| R_mask
    #mask = S_mask | Sx_mask #| R_mask


    ##"negative" threshold to reject artifacts?
    # roadside_mask = (img_HLS[:,:,0] < 40) & (img_HLS[:,:,2] > 30)  & (img_HLS[:,:,2] < 120)
    # asphalt_mask = (img_HLS[:,:,0] > 100) & (img_HLS[:,:,2] < 30)
    ##roadside_mask = (img_HLS[:,:,2] <= S_thd[0])
    # reject_mask = (~roadside_mask) & (~asphalt_mask)
#plot detailed analysis (useful to verify properties and select thresholds)


#define function to process mask using steps tested above
def lanepxmask(img_RGB):
    """ Take RGB image, perform necessary color transformation /gradient calculations
        and output the detected lane pixels mask, alongside an RGB composition of the 3 sub-masks (added)
        for visualization
    """
    # convert to HLS color space
    img_HLS = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HLS)
    # output ==> np.shape(img_HLS) = (Ny, Ny, 3 = [H, L, S])

    # high S value
    S_thd = (200, 255)
    S_mask = (img_HLS[:,:,2] > S_thd[0]) & (img_HLS[:,:,2] <= S_thd[1])
    # high S x-gradient
    S_gradx_mask = abs_sobel_thresh(img_HLS[:,:,2], orient='x', thresh=(20, 100),
                                   sobel_kernel=sobel_kernel, GRAY_INPUT=True)
    # high x-gradient of grayscale image (converted internally)
    gradx_mask = abs_sobel_thresh(img_RGB, orient='x', thresh=(20, 100), sobel_kernel=sobel_kernel)

    # build main lane pixel mask
    mask = S_gradx_mask | S_gradx_mask | S_mask
    # prepare RGB auxiliary for visualization:
    # stacked individual contributions in RGB = (S, gradx{Gray},  gradx{S})
    color_binary_mask = np.dstack((S_mask, gradx_mask, S_gradx_mask)) * 255

    return mask, color_binary_mask
# ------------------------------


"""III) Perspective Transformations and Region of Interest (ROI) """

class Warp2TopDown():
    """ Compute a perspective transform M, given source and destination points
        Use a class to provide a warping method and remember the points and relevant info as attributes
    """
    # -------------------------------
    def __init__(self):
        # based on a Ny x Nx = 720 x 1280 image (straight_lines1/2.jpg)
        self.pts_img = np.float32([[190 + 1, 720], [600 + 1, 445],
                              [680 - 2, 445], [1120 - 2, 720]])

        self.pts_warp = np.float32([[350, 720], [350, 0], [950, 0], [950, 720]])

        # direct perspective transform
        self.M = cv2.getPerspectiveTransform(self.pts_img, self.pts_warp)
        # inverse perspective transform:
        self.Minv = cv2.getPerspectiveTransform(self.pts_warp, self.pts_img)
    # -------------------------------
    def warp(self, img_in):
        # Warp an image using the perspective transform, M:
        Ny, Nx = np.shape(img_in)[0:2]
        img_warped = cv2.warpPerspective(img_RGB, self.M, (Nx, Ny), flags=cv2.INTER_LINEAR)
        return img_warped
    # -------------------------------
    def unwarp(self, img_in):
        # Inverse perspective transform, invM:
        Ny, Nx = np.shape(img_in)[0:2]
        img_unwarped = cv2.warpPerspective(img_RGB, self.invM, (Nx, Ny), flags=cv2.INTER_LINEAR)
        return img_unwarped
# -------------------------------

Perspective = Warp2TopDown() #object


# TODO: check if ROI is useful, make routine or incorporate!
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 600  # meters per pixel in x dimension

# define ROI for spatial filtering using inverse transform
# cf. equations at
# https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#getperspectivetransform
# auxiliar 3D coordinates are used with z =1 for source and a normalization factor t for the destination
# transposition is done to facilitate matrix product
# take the warp region as a reference and expand, to get a rectangle in the top-down view
xmin, xmax = Perspective.pts_warp[[0,2],0]
ymin, ymax = Perspective.pts_warp[[2,3],1]
pts_warpedROI = np.float32([[xmin-200, ymax, 1], [xmin-200, ymin, 1], [xmax+200, ymin, 1], [xmax+200, ymax, 1]]).T
pts_ROI = np.tensordot(Perspective.Minv.T, pts_warpedROI, axes=([0], [0]))
pts_ROI = (pts_ROI[0:2, :] / pts_ROI[2, :]).T
x_wROI, y_wROI = closePolygon(pts_warpedROI[0:2, :].T)
x_ROI, y_ROI = closePolygon(pts_ROI)


if SHOW_WARP:
    # make closed polygons for plotting
    x_img, y_img = closePolygon(Perspective.pts_img)
    x_warp, y_warp = closePolygon(Perspective.pts_warp)

    # do not use the plotNimg() routine, as additional annotations are needed
    fig = plt.figure(figsize=(12, 4.5), num=5)
    fig.canvas.set_window_title('Perspective Transform')
    plt.subplot(1, 2, 1)
    plt.imshow(img_RGB)
    plt.plot(x_img, y_img, 'r--')
    plt.plot(x_ROI, y_ROI, 'b')
    plt.title('Input Image')
    plt.subplot(1, 2, 2)
    plt.imshow(Perspective.warp(img_RGB))
    plt.plot(x_warp, y_warp, 'r--')
    plt.plot(x_wROI, y_wROI, 'b')
    plt.title('Warped "top-down" Image')
    plt.savefig(output_dir + img_basename + '_warp.png')


"""IV) Get restricted and warped detection masks"""

# A) detect before warp
if SHOW_COLOR_GRADIENT:
    # get mask and stacked individual contributions in RGB (S, Gray-gradx, S-gradx)
    mask, c_bin = lanepxmask(img_RGB)
    # restrict to ROI
    mask_inROI = restrict2ROI(255 * np.uint8(mask), pts_ROI)
    # warp detection mask
    mask_warp = Perspective.warp(mask_inROI)

    # plot the image-domain detection masks (separate by contribution), ROI-restricted masks and warped mask
    plotNimg([c_bin, mask, mask_inROI, mask_warp],
             ['Mask components', 'Output mask', 'ROI-restricted', 'Warped'],
             [None, "gray", "gray", "gray"],
             'Lane masks', fig_num=6)
    plt.savefig(output_dir + img_basename + '_masks-img.png')

# B) warp before detect ==> less blurry
#get warped RGB image
imgRGB_warped = Perspective.warp(img_RGB)
mask_warped, cbin_warped = lanepxmask(imgRGB_warped)


plotNimg([imgRGB_warped, cbin_warped, mask_warped], ['RGB image', 'Mask components', 'Output mask'],
             [None, None, "gray"], 'Warped Image and Detection mask', fig_num=7)
plt.savefig(output_dir +'1'+ img_basename + '_masks-warp.png')


"""V) Process mask and get polynomial fit for boundaries """

fig = plt.figure(num=10)
fig.canvas.set_window_title("Detected lanes: from pixel mask")
left, right, out_img = find_lane_xy_frommask(mask_warped)
print('')
print('Fit from pixel search')
print('Left fit: ', left.cf)
print('Right fit: ', right.cf)
plt.imshow(out_img)
plt.savefig(output_dir + '2' +img_basename + '_slidingwindowfit.png')

plt.figure(num=11)
gradx_mask = abs_sobel_thresh(imgRGB_warped, orient='x', thresh=(20, 100), sobel_kernel=sobel_kernel)
left, right, out_img = find_lane_xy_frommask(255*np.uint8(gradx_mask))
print('')
print('Alternative mask')
print('Left fit: ', left.cf)
print('Right fit: ', right.cf)
plt.imshow(out_img)
plt.savefig(output_dir + '20' +img_basename + '_slidingwindowfit2.png')


#find_lane_xy_frompoly(mask_warped, cf1, cf2)


# treatment of mask
kernel = np.ones((3, 3), np.uint8)
# remove noise by applying Morphological open?
mask_mostreliable = cv2.morphologyEx(255 * np.uint8(mask_warped), cv2.MORPH_OPEN, kernel, iterations=2)
# perform watershed detection (labels connected components with unique number
ret, labels = cv2.connectedComponents(mask_mostreliable)
# get indices which belong to each of the reliable clusters (will not be split by the margin)
# bins_map [j] contains indices of connected region j
bins_map = [np.where(labels == j)[0] for j in range(1, np.max(labels))]

