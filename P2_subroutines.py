# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:13:28 2019

@author: Felipe
"""

"""Collect sub-routines used as tools"""

""" Imports """
# modules

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2, os, io, glob
from PIL import Image
from pdb import set_trace as stop  # useful for debugging


"""------------------------------------------------------------"""
""" Image display                                              """
"""------------------------------------------------------------"""

# ---------------------------------------------------------------------
def weighted_img(img_annotation, initial_img, α=0.8, β=1., γ=0.):
    ###mixes two color images of the same shape, img and initial_img, for annotations
    return cv2.addWeighted(initial_img, α, img_annotation, β, γ)


# ---------------------------------------------------------------------
def restrict2ROI(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, np.int32([vertices]), ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# ---------------------------------------------------------------------
def plotNimg(imgs, titles, cmap, fig_title, fig_num=None):
    # plot a stack of images with given titles and color maps
    # cmap='gray' for single channel!
    N_img = len(imgs)
    N_row = int(np.ceil(N_img / 2.0))
    fig = plt.figure(num=fig_num, figsize=(12, 4.5 * N_row))
    fig.canvas.set_window_title(fig_title)

    for i_img in range(N_img):
        plt.subplot(N_row, 2, i_img + 1)
        plt.imshow(imgs[i_img], cmap=cmap[i_img])
        plt.title(titles[i_img])


# ---------------------------------------------------------------------
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


# ---------------------------------------------------------------------
def closePolygon(pts_xy):
    # get a poligon with [N, 2] vertices and returns x and y for plotting
    # by repeating the last point
    x = np.concatenate((pts_xy[:, 0], [pts_xy[0, 0]]))
    y = np.concatenate((pts_xy[:, 1], [pts_xy[0, 1]]))
    return x, y


# ---------------------------------------------------------------------
def get_plot(fignum=[]):
    # read contents of figure as numpy array and return it
    # set specified figure as current, if argument is provided
    if np.size(fignum) != 0:
        fig = plt.figure(num=fignum)
    buff = io.BytesIO()
    plt.savefig(buff, facecolor='w', edgecolor='w', format='png')
    buff.seek(0)
    return np.array(Image.open(buff).convert('RGB'))


# ---------------------------------------------------------------------


"""------------------------------------------------------------"""
""" Camera Calibration                                         """
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

        # Undistort a test images and save results:
        output_dir = 'output_images' + os.sep + 'calibration' + os.sep
        os.makedirs(output_dir, exist_ok=True)

        for img_path in cal_images:

            image = mpimg.imread(img_path)
            cal_image = cv2.undistort(image, cal_mtx, dist_coef, None, cal_mtx)
            img_basename = os.path.basename(img_path).split('.jpg')[0]

            # input image
            fig = plt.figure(num=1)
            plt.clf()
            fig.canvas.set_window_title('Input Image')
            plt.imshow(image)
            plt.savefig(output_dir + img_basename+".jpg", format='jpg')
            # calibrated image
            fig = plt.figure(num=2)
            plt.clf()
            fig.canvas.set_window_title('Input Image after Calibration')
            plt.imshow(cal_image)
            plt.savefig(output_dir + img_basename+"_output.jpg", format='jpg')

    plt.close('all')

    # return calibration matrix and distortion coefficients to be used with
    # cv2.undistort(image, cal_mtx, dist_coef, None, cal_mtx)
    return cal_mtx, dist_coef
# ------------------------------
def undistort(image, cal_mtx, dist_coef):
    # apply the above coefficients/matrix to a given image
    return cv2.undistort(image, cal_mtx, dist_coef, None, cal_mtx)
# ------------------------------


"""------------------------------------------------------------"""
""" Masking and Thresholding Image or Gradient                 """
"""------------------------------------------------------------"""

# 1) Thresholding tools, as developed in quizzes
# ---------------------------------------------------------------------
def abs_sobel_thresh(img, orient='y', thresh=(0, 255),
                     GRAY_INPUT=False, sobel_kernel=3):
    # Define a function that applies Sobel x or y,
    # then takes an absolute value and applies a threshold.
    # ---------------------------------------------------------------------

    # Apply the following steps to img
    # 1) Convert to grayscale
    if GRAY_INPUT == False:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # use cf2.CV_64F (double) to avoid problem with negative (white2black) derivatives!
    if orient in ['x', 'y']:  # x or y direction
        direction = {'x': (1, 0), 'y': (0, 1)}
        sobel_grad = cv2.Sobel(img_gray, cv2.CV_64F,
                               direction[orient][0], direction[orient][1],
                               ksize=2 * (sobel_kernel // 2) + 1)
    else:
        raise Exception('Unknown direction: ' + orient)
    # 3) Take the absolute value of the derivative or gradient
    sobel_grad = np.abs(sobel_grad)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel_byte = np.uint8(255 * sobel_grad / np.max(sobel_grad))

    # 5) Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    if thresh == (None, None):
        mask = sobel_byte  # return byte image (to check levels)
    else:
        mask = (sobel_byte > thresh[0]) & (sobel_byte < thresh[1])
    # 6) Return this mask as your binary_output image
    return mask
# ---------------------------------------------------------------------
def mag_thresh(img, sobel_kernel=3, thresh=(0, 255), GRAY_INPUT=False):
    # Define a function that applies Sobel x and y,
    # then computes the magnitude of the gradient
    # and applies a threshold
    # ---------------------------------------------------------------------
    # Apply the following steps to img
    # 1) Convert to grayscale
    if GRAY_INPUT == False:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    # 2) Take the gradient in x and y separately
    gradx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=2 * (sobel_kernel // 2) + 1)
    grady = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=2 * (sobel_kernel // 2) + 1)
    # 3) Calculate the magnitude 
    grad_mag = np.sqrt(gradx ** 2 + grady ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    grad_byte = np.uint8(255 * grad_mag / np.max(grad_mag))
    # 5) Create a binary mask where mag thresholds are met
    mask = (grad_byte > thresh[0]) & (grad_byte < thresh[1])
    # 6) Return this mask as your binary_output image
    return mask

# ---------------------------------------------------------------------
def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi / 2),
               GRAY_INPUT=False):
    # Define a function that applies Sobel x and y,
    # then computes the direction of the gradient
    # and applies a threshold.
    # ---------------------------------------------------------------------
    # Apply the following steps to img
    # 1) Convert to grayscale
    if GRAY_INPUT == False:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    # 2) Take the gradient in x and y separately
    gradx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=2 * (sobel_kernel // 2) + 1)
    grady = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=2 * (sobel_kernel // 2) + 1)
    # 3) Take the absolute value of the x and y gradients
    grad_absx = np.abs(gradx)
    grad_absy = np.abs(grady)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_dir = np.arctan2(grad_absy, grad_absx)

    # 5) Create a binary mask where direction thresholds are met    
    if thresh == (None, None):
        mask = grad_dir  # return byte image (to check levels)
    else:
        mask = (grad_dir > thresh[0]) & (grad_dir < thresh[1])
    # 6) Return this mask as your binary_output image
    return mask

# ---------------------------------------------------------------------
# 2) adopted mask processing strategy
# ---------------------------------------------------------------------
# define function to process mask using steps tested in the single frame pipeline
def lanepxmask(img_RGB, sobel_kernel=7):
    """ Take RGB image, perform necessary color transformation /gradient calculations
        and output the detected lane pixels mask, alongside an RGB composition of the 3 sub-masks (added)
        for visualization
    """
    MORPH_ENHANCE = True

    # morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # convert to HLS color space
    img_HLS = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HLS)
    # output ==> np.shape(img_HLS) = (Ny, Ny, 3 = [H, L, S])

    # high S value
    S_thd = (200, 255)
    S_mask = (img_HLS[:, :, 2] > S_thd[0]) & (img_HLS[:, :, 2] <= S_thd[1])
    # reject shadows from S mask
    S_mask[img_HLS[:,:,1] < 50] = False
    # CLOSE to close gaps
    if MORPH_ENHANCE:
        S_mask = cv2.morphologyEx(255 * np.uint8(S_mask), cv2.MORPH_CLOSE, kernel, iterations=2) > 0
        S_mask = cv2.dilate(255 * np.uint8(S_mask), kernel) > 0

    # high S x-gradient
    S_gradx_mask = abs_sobel_thresh(img_HLS[:, :, 2], orient='x', thresh=(20, 100),
                                    sobel_kernel=sobel_kernel, GRAY_INPUT=True)
    # reject shadows from S x-gracient mask
    S_mask[img_HLS[:,:,1] < 50] = False
    # CLOSE to close gaps
    if MORPH_ENHANCE:
        S_gradx_mask = cv2.morphologyEx(255 * np.uint8(S_gradx_mask), cv2.MORPH_CLOSE, kernel, iterations=2) > 0
        S_gradx_mask = cv2.dilate(255 * np.uint8(S_gradx_mask), kernel) > 0

    # high x-gradient of grayscale image (converted internally)
    gradx_mask = abs_sobel_thresh(img_RGB, orient='x', thresh=(20, 100), sobel_kernel=sobel_kernel)
    # reject dark lines from x-grad mask
    dark_borders = cv2.blur(img_HLS[:, :, 1], (15,15))<cv2.blur(img_HLS[:, :, 1], (17,17)) #lower L than vicinity
    dark_borders = cv2.morphologyEx(255*np.uint8(dark_borders), cv2.MORPH_OPEN, kernel, iterations=2)
    # overall dark region
    dark_regions = cv2.morphologyEx(255 * np.uint8(img_HLS[:,:,1] < (np.mean(img_HLS[:,:,1]))), cv2.MORPH_OPEN, kernel, iterations=2)
    # remove noise and dilate to block the border from appearing in gradient
    dark = cv2.morphologyEx(255 * np.uint8((dark_borders == 255) | (dark_regions == 255)), cv2.MORPH_OPEN, kernel, iterations=3)
    dark = cv2.dilate(dark,np.ones([sobel_kernel, sobel_kernel]), iterations=2)
    # sanity check: no white points removed
    white = cv2.inRange(gaussian_blur(img_RGB, 5), np.uint8([180, 180, 180]), np.uint8([255, 255, 255]))
    #yellow = cv2.inRange(gaussian_blur(img_RGB, 5), np.uint8([160, 160, 120]), np.uint8([255, 255, 0]))
    yellow = (img_RGB[:, :, 0] > 160) & (img_RGB[:, :, 1] > 160) & (img_RGB[:, :, 2] < 120)
    dark[white == 255] = 0 # do not remove
    # remove dark border from gradient
    gradx_mask[dark == 255] = False
    # OPEN for noise reduction
    if MORPH_ENHANCE:
        gradx_mask = cv2.morphologyEx(255 * np.uint8(gradx_mask), cv2.MORPH_OPEN, kernel, iterations=2) > 0

    # propagate changes to S-based masks, enhance white and yellow
    S_mask[white == 255] = True
    S_mask[yellow] = True
    S_mask[dark_regions == 255] = False
    S_gradx_mask[dark_regions == 255] = False

    # build main lane pixel mask
    mask = S_gradx_mask | gradx_mask | S_mask

    # prepare RGB auxiliary for visualization:
    # stacked individual contributions in RGB = (S, gradx{Gray},  gradx{S})
    color_binary_mask = np.dstack((S_mask, gradx_mask, S_gradx_mask)) * 255

    return mask, color_binary_mask


# ------------------------------


"""------------------------------------------------------------"""
""" Warping and Perspective Transformation                     """
"""------------------------------------------------------------"""
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
        img_warped = cv2.warpPerspective(img_in, self.M, (Nx, Ny), flags=cv2.INTER_LINEAR)
        return img_warped

    # -------------------------------
    def unwarp(self, img_in):
        # Inverse perspective transform, invM:
        Ny, Nx = np.shape(img_in)[0:2]
        img_unwarped = cv2.warpPerspective(img_in, self.Minv, (Nx, Ny), flags=cv2.INTER_LINEAR)
        return img_unwarped

    # -------------------------------
    # define ROI for spatial filtering using inverse transform
    # cf. equations at
    # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#getperspectivetransform
    # auxiliar 3D coordinates are used with z =1 for source and a normalization factor t for the destination
    # transposition is done to facilitate matrix product
    # take the warp region as a reference and expand, to get a rectangle in the top-down view representing relevant
    # search area
    # -------------------------------
    def xy_ROI_img(self):
        xmin, xmax = self.pts_warp[[0, 2], 0]
        ymin, ymax = self.pts_warp[[2, 3], 1]
        pts_warpedROI = np.float32([[xmin - 200, ymax, 1],
                                    [xmin - 200, ymin, 1],
                                    [xmax + 200, ymin, 1],
                                    [xmax + 200, ymax, 1]]).T
        x_wROI, y_wROI = closePolygon(pts_warpedROI[0:2, :].T)  # in warped domain
        pts_ROI = np.tensordot(self.Minv.T, pts_warpedROI, axes=([0], [0]))
        pts_ROI = (pts_ROI[0:2, :] / pts_ROI[2, :]).T
        x_ROI, y_ROI = closePolygon(pts_ROI)  # in image domain
        return x_ROI, y_ROI, pts_ROI

    # -------------------------------
    def xy_ROI_warped(self):
        xmin, xmax = self.pts_warp[[0, 2], 0]
        ymin, ymax = self.pts_warp[[2, 3], 1]
        pts_warpedROI = np.float32([[xmin - 200, ymax, 1],
                                    [xmin - 200, ymin, 1],
                                    [xmax + 200, ymin, 1],
                                    [xmax + 200, ymax, 1]]).T
        x_wROI, y_wROI = closePolygon(pts_warpedROI[0:2, :].T)  # in warped domain
        return x_wROI, y_wROI


# -------------------------------

"""------------------------------------------------------------"""
""" Mask processing and fitting                                """
"""------------------------------------------------------------"""

# ---------------------------------------------------------------------
def color_preprocessing(img_RGB, GET_BOX=False):
    """ Apply color-based pre-processing of frames"""
    box_size = 30
    box_ystep = 80
    box_vertices = box_size * np.array([(-1, -1),  # bottom left
                                        (-1, 1), (1, 1),
                                        (1, -1)], dtype=np.int32)  # bottom right

    x_box, y_box = closePolygon(box_vertices)
    # get average color of boxes and mask out
    image_new = np.copy(img_RGB)
    Ny, Nx = np.shape(img_RGB)[0:2]
    for i_box in range(3):
        box = img_RGB[Ny - i_box * box_ystep - 2 * box_size:Ny - i_box * box_ystep,
              Nx // 2 - box_size:Nx // 2 + box_size, :]
        avg_color = np.mean(box, axis=(0, 1))
        mask = cv2.inRange(gaussian_blur(img_RGB, 7), np.uint8(avg_color - 25), np.uint8(avg_color + 25))
        image_new = cv2.bitwise_and(image_new, image_new, mask=255 - mask)  # delete (mask out) similar colors

    # remove black artifacts
    mask = cv2.inRange(gaussian_blur(img_RGB, 7), np.uint8([0, 0, 0]), np.uint8([50, 50, 50]))
    image_new = cv2.bitwise_and(image_new, image_new, mask=255 - mask)  # delete (mask out) similar colors

    # remove gray artifacts
    mask = cv2.inRange(gaussian_blur(img_RGB, 7), np.uint8([80, 80, 80]), np.uint8([120, 120, 120]))
    image_new = cv2.bitwise_and(image_new, image_new, mask=255 - mask)  # delete (mask out) similar colors


    if GET_BOX: #return image and box parameters for plotting
        return image_new, x_box, y_box, box_size, box_ystep
    else:
        return image_new
# ---------------------------------------------------------------------



# functions to process the mask by finding lane pixels and fitting
# (based on quizzes)
# ---------------------------------------------------------------------
class LaneLine:
    """store the coordinates of the pixels and the polynomial coefficients for each lane
     also store where the line reaches the bottom of the image """

    def __init__(self, x_coord, y_coord, poly_coef, MSE, y_min_reliable):
        self.Npix = np.size(x_coord) #size of fitted region
        self.x_pix = x_coord
        self.y_pix = y_coord
        self.cf = poly_coef
        if np.size(y_coord) > 0:
            self.x_bottom = np.polyval(poly_coef, np.max(y_coord))
        else:
            self.x_bottom = None
        # MSE of fit (compare "goodness of fit")
        self.MSE = MSE
        if np.isnan(y_min_reliable):
            self.y_min_reliable = None #neutral indexing value
        else:
            self.y_min_reliable = np.int32(y_min_reliable)
# ---------------------------------------------------------------------
def weight_fit_cfs(left, right):
    """ judge fit quality, providing weights and a weighted average of the coefficients
        inputs are LaneLine objects """
    cfs = np.vstack((left.cf, right.cf))
    cf_MSE = np.vstack((left.MSE, right.MSE))
    # average a/b coefficients with inverse-MSE weights
    w1 = np.sum(cf_MSE) / cf_MSE
    # consider number of points as well
    w2 = np.reshape(np.array([left.Npix, right.Npix]) / (left.Npix + right.Npix), [2, 1])
    # aggregate weights
    w = w1 * w2
    cf_avg = np.mean(w * cfs, axis=0) / np.mean(w, axis=0)
    return w, cf_avg
# ---------------------------------------------------------------------
def find_lane_xy_frommask(mask_input, nwindows=9, margin=100, minpix=50, NO_IMG=False):
    """ Take the input mask and perform a sliding window search
     Return the coordinates of the located pixels, polynomial coefficients and optionally an image showing the
       windows/detections
     **Parameters/Keywords:
     nwindows ==> Choose the number of sliding windows
     margin ==> Set the width of the windows +/- margin
     minpix ==> Set minimum number of pixels found to recenter window
     NO_IMG ==> do not calculate the diagnose output image (used in pipeline)"""

    # convert to byte
    mask_input = 255 * np.uint8(mask_input / np.max(mask_input))

    # treatment of mask to get more reliable region
    kernel = np.ones((3, 3), np.uint8)
    # remove noise by applying Morphological open?
    mask_mostreliable = cv2.morphologyEx(255 * np.uint8(mask_input), cv2.MORPH_OPEN, kernel, iterations=2)
    # perform watershed detection (labels connected components with unique number
    ret, labels = cv2.connectedComponents(mask_mostreliable)
    # get indices which belong to each of the reliable clusters (will not be split by the margin)
    # bins_map [j-1] contains lin/col = yx = [0,1] indices of connected region j ([2, Npts])
    # take indices of nonzero (defined in line 267 below) so that the indices refer to the same mask
    region_idx_map = [(labels == j).nonzero() for j in range(1, np.max(labels) + 1)]

    # Take a "histogram" (x-profile) of the bottom half of the image
    histogram = np.sum(mask_input[mask_input.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    # Consider a margin to avoid locating lanes too close to border
    Ny, Nx = mask_input.shape[0:2]
    midpoint = np.int(Nx // 2)
    leftx_base = np.argmax(histogram[Nx // 10:midpoint]) + Nx // 10
    rightx_base = np.argmax(histogram[midpoint:Nx - Nx // 10]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(Ny // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = mask_input.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices/coordinates (first is x)
    # inds ==> refer to the image, not nonzero!
    left_lane_inds  = [[], []]
    right_lane_inds = [[], []]

    # Create an output image to draw on and visualize the result
    if NO_IMG == False:
        out_img = np.dstack((mask_input, mask_input, mask_input))
    else:
        out_img = None  # no image returned

    # keep track of the minimumy of the reliable (label-based regions)
    ymin_good_left = np.nan
    ymin_good_right = np.nan

    # Step through the windows one by one
    for window in range(nwindows):

        # Identify window boundaries in x and y (and right and left)
        win_y_low = Ny - (window + 1) * window_height
        win_y_high = Ny - window * window_height
        # Find the four boundaries of the window #
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        if NO_IMG == False:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        # take indices of the non-zero selection!
        good_left_inds = np.where((nonzerox >= win_xleft_low) & (nonzerox <= win_xleft_high) &
                                  (nonzeroy >= win_y_low) & (nonzeroy <= win_y_high))[0]
        good_right_inds = np.where((nonzerox >= win_xright_low) & (nonzerox <= win_xright_high) &
                                   (nonzeroy >= win_y_low) & (nonzeroy <= win_y_high))[0]

        # Append these coordinates to the lists
        # left lane
        left_lane_inds[0].append(nonzerox[good_left_inds])
        left_lane_inds[1].append(nonzeroy[good_left_inds])
        # right lane
        right_lane_inds[0].append(nonzerox[good_right_inds])
        right_lane_inds[1].append(nonzeroy[good_right_inds])

        # check the connected regions inside the selection
        # if points are found, add the whole region to the selection of good indices
        # kept a separate list as this refers to the image indices (and not nonzero, as the labeling
        # requires the full image and not only the non zero points of the mask)
        labels_in_left = np.unique(labels[nonzeroy[good_left_inds], nonzerox[good_left_inds]])
        labels_in_left = labels_in_left[labels_in_left > 0]  # 0 = background
        if np.size(labels_in_left) > 0:
            for k in labels_in_left:
                # y indices of the whole region ==> get portion of region within y-window
                # value of label k maps to index k-1!
                yreg_left, xreg_left = region_idx_map[k - 1][0], region_idx_map[k - 1][1]
                reg_good_idx = np.where((yreg_left >= win_y_low) & (yreg_left <= win_y_high))[0]
                # store x and y coordinates in the appropriate lists
                left_lane_inds[0].append(xreg_left[reg_good_idx])
                left_lane_inds[1].append(yreg_left[reg_good_idx])
                # keep track of minimum value
                ymin_good_left = np.nanmin(np.concatenate(([ymin_good_left], yreg_left[reg_good_idx])))

        # same for right
        labels_in_right = np.unique(labels[nonzeroy[good_right_inds], nonzerox[good_right_inds]])
        labels_in_right = labels_in_right[labels_in_right > 0]  # 0 = background
        if np.size(labels_in_right) > 0:
            for k in labels_in_right:
                # y indices of the whole region ==> get portion of region within y-window
                # value of label k maps to index k-1!
                yreg_right, xreg_right = region_idx_map[k - 1][0], region_idx_map[k - 1][1]
                reg_good_idx = np.where((yreg_right >= win_y_low) & (yreg_right <= win_y_high))[0]
                # store x and y coordinates in the appropriate lists
                right_lane_inds[0].append(xreg_right[reg_good_idx])
                right_lane_inds[1].append(yreg_right[reg_good_idx])
                # keep track of minimum value
                ymin_good_right = np.nanmin(np.concatenate(([ymin_good_right], yreg_right[reg_good_idx])))


        # Update window's x center
        # left window
        # perform a fit to predict the window tendency, if a minimum number of points is present and the y span allows
        if (np.size(np.concatenate(left_lane_inds[1])) >= minpix) and \
                (np.max(np.concatenate(left_lane_inds[1]))- np.min(np.concatenate(left_lane_inds[1]))) > minpix:
            # make partial fit of order 1 or 2
            order = 1*(window<3) + 2*(window>=3)
            polycf_left = np.polyfit(np.concatenate(left_lane_inds[1]),
                                     np.concatenate(left_lane_inds[0]), order)
            # predict position at next window
            leftx_current = np.int(np.round(np.polyval(polycf_left, 0.5 * (win_y_low + win_y_high)-window_height)))

        # right window
        # perform a fit to predict the window tendency, if a minimum number of points is present and the y span allows
        if (np.size(np.concatenate(right_lane_inds[1])) >= minpix) and \
                (np.max(np.concatenate(right_lane_inds[1]))- np.min(np.concatenate(right_lane_inds[1]))) > minpix:
            # make partial fit of order 1 or 2
            order = 1*(window < 3) + 2*(window>=3)
            polycf_right = np.polyfit(np.concatenate(right_lane_inds[1]),
                                      np.concatenate(right_lane_inds[0]), order)
            # predict position at next window
            rightx_current = np.int(np.round(np.polyval(polycf_right, 0.5 * (win_y_low + win_y_high) - window_height)))

        # keep this plot ==> cool to explain procedure!
        PLOT_METHOD = False
        if PLOT_METHOD and (window == 4):
            stop()
            plt.figure(num=1)
            plt.clf()
            plt.imshow(out_img)
            y = np.arange(win_y_low-window_height, win_y_high)
            plt.plot(np.polyval(polycf_left, y), y, 'r--')
            plt.plot(np.repeat(leftx_current, 2), np.repeat(0.5 * (win_y_low + win_y_high) - window_height,2), 'b+')
            plt.plot(np.polyval(polycf_right, y), y, 'r--')
            plt.plot(np.repeat(rightx_current, 2), np.repeat(0.5 * (win_y_low + win_y_high) - window_height,2), 'b+')
            plt.pause(20)
            stop()


    # for each window

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    # Extract left and right line pixel positions
    # all indices refer to the whole mask!
    leftx = np.concatenate(left_lane_inds[0])
    lefty = np.concatenate(left_lane_inds[1])
    rightx = np.concatenate(right_lane_inds[0])
    righty = np.concatenate(right_lane_inds[1])

    # Fit a second order polynomial to each using `np.polyfit`, assuming x = f(y) #
    # for info on residuals ==> https: // stackoverflow.com / questions / 5477359 / chi - square - numpy - polyfit - numpy
    # use weights based on y, so the bottom pixels are weighted more
    polycf_left, sqr_error_left, _, _, _ = np.polyfit(lefty, leftx, 2, full = True, w=lefty / Ny)
    MSE_left = np.sqrt(sqr_error_left[0] / np.size(lefty))

    polycf_right, sqr_error_right, _, _, _ = np.polyfit(righty, rightx, 2, full = True, w=righty / Ny)
    MSE_right = np.sqrt(sqr_error_right[0] / np.size(righty))

    # Lane annotation (to be warped and shown in pipeline)
    lane_annotation = np.zeros([Ny, Nx, 3], dtype=np.uint8)
    lane_annotation[lefty, leftx] = [255, 0, 0]
    lane_annotation[righty, rightx] = [0, 0, 255]

    # Optional Visualization Steps
    if NO_IMG == False:
        # Set colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Generate x and y values for plotting
        ploty = np.linspace(0, Ny - 1, Ny)

        left_fitx = np.polyval(polycf_left, ploty)
        right_fitx = np.polyval(polycf_right, ploty)

        # Plots the left and right polynomials on the lane lines (only works in non-interactive backend!)
        # otherwise ends up in currently open figure
        if plt.get_backend() == 'Agg':
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
    # Optional Visualization Steps

    # wrap data into LaneLines objects
    leftlane = LaneLine(leftx, lefty, polycf_left, MSE_left, ymin_good_left)
    rightlane = LaneLine(rightx, righty, polycf_right, MSE_right, ymin_good_right)

    return leftlane, rightlane, lane_annotation, out_img


# ---------------------------------------------------------------------
def find_lane_xy_frompoly(mask_input, polycf_left, polycf_right, margin=80, NO_IMG=False):
    """ Take the input mask and perform a search around the polynomial-matching area
     Return the coordinates of the located pixels, polynomial coefficients and optionally an image showing the
       windows/detections
     **Parameters/Keywords:
     margin ==> Set the width of the windows +/- margin
     NO_IMG ==> do not calculate the diagnose output image (used in pipeline) """

    # convert bool to byte
    mask_input = 255 * np.uint8(mask_input / np.max(mask_input))
    Ny, Nx = np.shape(mask_input)

    # treatment of mask to get more reliable region
    kernel = np.ones((3, 3), np.uint8)
    # remove noise by applying Morphological open?
    mask_mostreliable = cv2.morphologyEx(255 * np.uint8(mask_input), cv2.MORPH_OPEN, kernel, iterations=2)
    # perform watershed detection (labels connected components with unique number
    ret, labels = cv2.connectedComponents(mask_mostreliable)
    # get indices which belong to each of the reliable clusters (will not be split by the margin)
    # bins_map [j-1] contains lin/col = yx = [0,1] indices of connected region j ([2, Npts])
    # take indices of nonzero (defined in line 267 below) so that the indices refer to the same mask
    region_idx_map = [(labels == j).nonzero() for j in range(1, np.max(labels) + 1)]

    # Locate activated pixels
    nonzero = mask_input.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the area of search based on activated x-values #
    # within the +/- margin of our polynomial function #
    x_nonzeropoly_left = np.polyval(polycf_left, nonzeroy)
    left_lane_inds = ((nonzerox > x_nonzeropoly_left - margin) &
                      (nonzerox < x_nonzeropoly_left + margin))
    x_nonzeropoly_right = np.polyval(polycf_right, nonzeroy)
    right_lane_inds = ((nonzerox > x_nonzeropoly_right - margin) &
                       (nonzerox < x_nonzeropoly_right + margin))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]


    lane_annotation = np.zeros([Ny, Nx, 3], dtype=np.uint8)
    lane_annotation[lefty, leftx] = [255, 0, 0]
    lane_annotation[righty, rightx] = [0, 0, 255]

    # check the connected regions inside the selection
    # if points are found, add the whole region to the selection of good indices
    # indices refer to the image indices (and not nonzero, as the labeling requires the full image
    # and not only the non zero points of the mask)
    leftx_fromlabels , lefty_fromlabels = [], []
    labels_in_left = np.unique(labels[lefty, leftx])
    labels_in_left = labels_in_left[labels_in_left > 0]  # 0 = background
    if np.size(labels_in_left) > 0:
        for k in labels_in_left:
            # xy indices of the whole region ==> get portion of region within y-window
            # value of label k maps to index k-1!
            yreg, xreg = region_idx_map[k - 1][0], region_idx_map[k - 1][1]
            leftx_fromlabels.append(xreg)
            lefty_fromlabels.append(yreg)
        #add to indices for fitting
        leftx_fromlabels = np.concatenate(leftx_fromlabels)
        lefty_fromlabels = np.concatenate(lefty_fromlabels)
        leftx = np.concatenate((leftx, leftx_fromlabels))
        lefty = np.concatenate((lefty, lefty_fromlabels))
        ymin_good_left = np.min(lefty_fromlabels)
    else:
        ymin_good_left = np.nan

    # right side
    rightx_fromlabels, righty_fromlabels = [], []
    labels_in_right = np.unique(labels[righty, rightx])
    labels_in_right = labels_in_right[labels_in_right > 0]  # 0 = background
    if np.size(labels_in_right) > 0:
        for k in labels_in_right:
            # xy indices of the whole region ==> get portion of region within y-window
            # value of label k maps to index k-1!
            yreg, xreg = region_idx_map[k - 1][0], region_idx_map[k - 1][1]
            rightx_fromlabels.append(xreg)
            righty_fromlabels.append(yreg)
            # add to indices for fitting
        rightx_fromlabels = np.concatenate(rightx_fromlabels)
        righty_fromlabels = np.concatenate(righty_fromlabels)
        rightx = np.concatenate((rightx, rightx_fromlabels))
        righty = np.concatenate((righty, righty_fromlabels))
        ymin_good_right = np.min(righty_fromlabels)
    else:
        ymin_good_right = np.nan

    # Fit new polynomial
    # Fit a second order polynomial to each with np.polyfit() : x = f(y)
    # Estimate goodness of fit by MSE
    polycf_left, sqr_error_left, _, _, _ = np.polyfit(lefty, leftx, 2, full=True, w=lefty / Ny)
    MSE_left = np.sqrt(sqr_error_left[0] / np.size(lefty))

    polycf_right, sqr_error_right, _, _, _ = np.polyfit(righty, rightx, 2, full=True, w=righty / Ny)
    MSE_right = np.sqrt(sqr_error_right[0] / np.size(righty))

    # Generate x and y values for plotting
    ploty = np.linspace(0, Ny - 1, Ny)
    # Evaluate both polynomials using ploty, polycf_left and polycf_right
    left_fitx = np.polyval(polycf_left, ploty)
    right_fitx = np.polyval(polycf_right, ploty)

    # Lane annotation (to be warped and shown in pipeline)
    lane_annotation = np.zeros([Ny, Nx, 3], dtype=np.uint8)
    lane_annotation[lefty, leftx] = [255, 0, 0]
    lane_annotation[righty, rightx] = [0, 0, 255]

    # Optional Visualization Steps
    if NO_IMG == False:
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((mask_input, mask_input, mask_input)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Plot the polynomial lines onto the image
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
    # Optional Visualization Steps
    else:
        out_img = None

    # wrap data into LaneLines objects
    leftlane = LaneLine(leftx, lefty, polycf_left, MSE_left, ymin_good_left)
    rightlane = LaneLine(rightx, righty, polycf_right, MSE_right, ymin_good_right)

    return leftlane, rightlane, lane_annotation, out_img


# -------------------------------------------------------------------
def getlane_annotation(mask_shape, polycf_left, polycf_right, img2annotate=[],
                       xmargin=5, ymin=0, PLOT_LINES=False):
    """ Give the shape and the polynomial coefficients, return byte mask showing the region inside the two curves
        (plus an optional x-margin). Also add the annotations to an image if it is provided.
     **Parameters/Keywords:
     img2annotate ==> RGB image to annotate, if provided (should match the size of mask_shape!)
     xmargin ==> Set the width of the +/- margin "
     PLOT_LINES ==> Add the polynomial lines to output (only relevant if a img2annnotate is present)"""

    # get sizes and declare output
    Ny, Nx = mask_shape[0:2]
    out_mask = np.zeros([Ny, Nx], dtype=np.uint8)

    if np.size(img2annotate) != 0:
        if np.shape(img2annotate)[0:2] != (Ny, Nx):  # note: must be tuple !
            raise Exception("Error: Image to annotate must match mask size!")
        # blank RGB to add lane are to
        img_annotated = np.zeros_like(img2annotate)
    else:
        img_annotated = None
    # check if annotation is possible

    ploty = np.arange(ymin, Ny)  # y values to plot
    # Evaluate both polynomials using ploty, polycf_left and polycf_right
    left_fitx = np.polyval(polycf_left, ploty)
    right_fitx = np.polyval(polycf_right, ploty)

    # Generate a polygon to illustrate the lane area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_lane = np.array([np.transpose(np.vstack([left_fitx - xmargin, ploty]))])
    right_line_lane = np.array([np.flipud(np.transpose(np.vstack([right_fitx + xmargin,
                                                                  ploty])))])
    lane_pts = np.hstack((left_line_lane, right_line_lane))

    # Draw the lane onto the mask
    cv2.fillPoly(out_mask, np.int_([lane_pts]), (255))

    # Draw the lane onto the warped blank image
    if np.size(img2annotate) != 0:
        cv2.fillPoly(img_annotated, np.int_([lane_pts]), (0, 255, 0))  # green lane
        img_annotated = cv2.addWeighted(img2annotate, 1, img_annotated, 0.3, 0)  # add to image
        # Plot the polynomial lines onto the image?
        if PLOT_LINES:
            plt.plot(left_fitx, ploty, color='yellow', linestyle='--')
            plt.plot(right_fitx, ploty, color='yellow', linestyle='--')
    # annotations

    return out_mask, img_annotated


# -------------------------------------------------------------------
def cf_px2m(poly_cf_px, img_shape):
    """ Convert from pixel polynomial coefficients (order 2) to m
        x = f(y), with origin at center/bottom of image, positive y upwards!"""
    Ny, Nx = img_shape[0:2]
    # Define conversions in x and y from pixels space to meters (approximate values for camera)
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # coordinate system of the lane [m] in top-down view:
    # x = 0 at the center of the image, +x = right
    # y = 0 at the botton, +y = top
    # pixel coordinate system: [0,0] at top left, [Nx, Ny] at bottom right
    # x_m = (x_px - Nx/2)*xm_per_pix
    # y_m = (Ny - y_px)*ym_per_pix
    a, b, c = poly_cf_px
    poly_cf_m = np.array([xm_per_pix / (ym_per_pix ** 2) * a,
                          -(2 * xm_per_pix / ym_per_pix * Ny * a + xm_per_pix / ym_per_pix * b),
                          xm_per_pix * (c - Nx / 2 + Ny * (b + a * Ny))])
    return poly_cf_m


# -------------------------------------------------------------------
# compute radius of curvature of a x=f(y) parabola, usually taken at bottom of image
def r_curve(polycoef, y):
    A, B = polycoef[0:2]
    return ((1 + (2 * A * y + B) ** 2) ** 1.5 / np.abs(2 * A))
# -------------------------------------------------------------------


"""------------------------------------------------------------"""
""" Frame Tracking                                             """
"""------------------------------------------------------------"""

# ---------------------------------------------------------------------
class LaneLine4Video():
    """ Define a helper class to receive the characteristics of each line detection (left/right)
        This container class gathers information across several frames, unlike LaneLines.
    """
    # -------------------------------
    def __init__(self, N_buffer = 10):
        # TODO: remove what is not used
        self.N_buffer = N_buffer

        # xy values
        # x values of the last N_buffer fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last N_buffer iterations
        self.bestx = None
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        self.y_min = 0 #last frame's y_min

        # coefficients
        # buffer with n last values
        self.cfs_buffer = np.zeros([N_buffer, 3], dtype='float')
        self.cfs_buffer[:] = np.nan  #invalid values
        #polynomial coefficients averaged over the last n iterations
        self.cfs_avg = np.zeros([3], dtype='float')
        #polynomial coefficients for the most recent fit
        self.cfs_current = np.zeros([3], dtype='float')
        #difference in fit coefficients between last and new fits
        self.cfs_diffs = np.zeros([3], dtype='float')
        # all-time statistics (for outlier detection ?)
        self.N_frames = 0
        self.cfs_avg_alltime = np.zeros([3], dtype='float')
        self.cfs_std_alltime = np.zeros([3], dtype='float')
        # store variance of fit coefficients (output of polyfit) to judge reliability
        self.cfs_uncertainty_buffer = np.zeros([N_buffer, 2], dtype='float')
        self.cfs_uncertainty_buffer[:] = np.nan  # invalid values
        self.cfs_uncertainty_avg = np.zeros([2], dtype='float')
        self.cfs_uncertainty_std = np.zeros([2], dtype='float')


        #curvature and lane center offset
        #radius of curvature of the line in m
        self.radius_of_curvature = None
        self.rcurve_buffer = np.zeros([N_buffer], dtype='float')
        self.rcurve_buffer[:] = np.nan #mark as invalid
        #distance in meters of vehicle center from the line
        self.line_center_offset = None
        self.lcenter_buffer = np.zeros([N_buffer], dtype='float')
        self.lcenter_buffer[:] = np.nan #mark as invalid

        # was the line detected in the last iteration?
        self.detected = False

    # -------------------------------
    # methods
    # -------------------------------
    def update_stats(self, new_cfs, fit_MSE, fit_Npoints):
        #update fields once a new set of coefficients is found
        # get new all-time average and standard deviation
        # https: // en.wikipedia.org / wiki / Algorithms_for_calculating_variance  # Welford's_online_algorithm
        N_prev = self.N_frames
        self.N_frames = self.N_frames +1
        mean_new = self.cfs_avg_alltime + np.float64(1) / (N_prev + 1) * (new_cfs - self.cfs_avg_alltime)
        self.cfs_avg_alltime =  np.sqrt( ((N_prev) * self.cfs_avg_alltime ** 2 +
                                          np.sum((new_cfs - self.cfs_avg_alltime) * (new_cfs - self.cfs_avg_alltime ))
                                          ) / (N_prev + 1) )
        self.cfs_avg_alltime = mean_new

        # save difference w.r.t. previous state
        self.cfs_diffs = new_cfs - self.cfs_current

        # overwrite oldest (ring buffer)
        self.cfs_buffer = np.roll(self.cfs_buffer, 1, axis=0)
        self.cfs_buffer[0, :] = new_cfs
        self.cfs_avg = np.nanmean(self.cfs_buffer, axis=0)
        self.cfs_median = np.nanmedian(self.cfs_buffer, axis=0)

        # keep uncertainty-related parameters
        self.cfs_uncertainty_buffer = np.roll(self.cfs_uncertainty_buffer, 1, axis=0)
        new_cf_uncertainty = [fit_MSE, fit_Npoints] #store MSE and number of points contributing to fit
        self.cfs_uncertainty_buffer[0, :] = new_cf_uncertainty
        self.cfs_uncertainty_avg = np.nanmean(self.cfs_uncertainty_buffer, axis=0)
        self.cfs_uncertainty_std = np.nanstd(self.cfs_uncertainty_buffer, axis=0)

        self.cfs_current = new_cfs
    # -------------------------------
    def update_xy(self, x_new, y_new, min_y_reliable):
        # update list of x values
        if len(self.recent_xfitted) == self.N_buffer: #full buffer
            oldest = self.recent_xfitted.pop(0) #this removes this entry
        self.recent_xfitted.append(x_new)
        # bookkeeping operations
        self.allx = x_new
        self.ally = y_new
        self.y_min = min_y_reliable
    # -------------------------------
    def update_R_coords(self, r_curve, line_center_offset):
        # rationale: keep a buffer to stabilize the values which are displayed
        # radius of curvature
        self.rcurve_buffer = np.roll(self.rcurve_buffer,1)
        self.rcurve_buffer[0] = r_curve
        self.radius_of_curvature = np.nanmedian(self.rcurve_buffer)
        # distance in meters of vehicle center from the line
        self.lcenter_buffer = np.roll(self.lcenter_buffer,1)
        self.lcenter_buffer[0] = line_center_offset
        self.line_center_offset = np.nanmedian(self.lcenter_buffer)
# ---------------------------------------------------------------------
