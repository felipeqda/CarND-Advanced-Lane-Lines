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
import cv2, os
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
def closePolygon(pts_xy):
    # get a poligon with [N, 2] vertices and returns x and y for plotting
    # by repeating the last point
    x = np.concatenate((pts_xy[:, 0], [pts_xy[0, 0]]))
    y = np.concatenate((pts_xy[:, 1], [pts_xy[0, 1]]))
    return x, y
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
    def xy_ROI_img(self):
        xmin, xmax = self.pts_warp[[0, 2], 0]
        ymin, ymax = self.pts_warp[[2, 3], 1]
        pts_warpedROI = np.float32( [[xmin - 200, ymax, 1],
                                     [xmin - 200, ymin, 1],
                                     [xmax + 200, ymin, 1],
                                     [xmax + 200, ymax, 1]]).T
        pts_ROI = np.tensordot(Perspective.Minv.T, pts_warpedROI, axes=([0], [0]))
        pts_ROI = (pts_ROI[0:2, :] / pts_ROI[2, :]).T
        x_wROI, y_wROI = closePolygon(pts_warpedROI[0:2, :].T) #in warped domain
        x_ROI, y_ROI = closePolygon(pts_ROI) #in image domain
        return x_ROI, y_ROI
# -------------------------------

"""------------------------------------------------------------"""
""" Mask processing and fitting                                """
"""------------------------------------------------------------"""
#functions to process the mask by finding lane pixels and fitting
#(based on quizzes)
# ---------------------------------------------------------------------
class LaneLine:
    """store the coordinates of the pixels and the polynomial coefficients for each lane
     also store where the line reaches the bottom of the image """
    def __init__(self, x_coord, y_coord, poly_coef):
        self.x_pix = x_coord
        self.y_pix = y_coord
        self.cf = poly_coef
        self.x_bottom = np.polyval(poly_coef, np.max(y_coord))
# ---------------------------------------------------------------------
def find_lane_xy_frommask(mask_input, nwindows = 9, margin = 100, minpix = 50, NO_IMG = False):
    """ Take the input mask and perform a sliding window search
     Return the coordinates of the located pixels, polynomial coefficients and optionally an image showing the
       windows/detections
     **Parameters/Keywords:
     nwindows ==> Choose the number of sliding windows
     margin ==> Set the width of the windows +/- margin
     minpix ==> Set minimum number of pixels found to recenter window
     NO_IMG ==> do not calculate the diagnose output image (used in pipeline)"""

    #convert to byte
    mask_input = 255*np.uint8(mask_input/np.max(mask_input))

    # treatment of mask
    kernel = np.ones((3, 3), np.uint8)
    # remove noise by applying Morphological open?
    mask_mostreliable = cv2.morphologyEx(255 * np.uint8(mask_input), cv2.MORPH_OPEN, kernel, iterations=2)
    # perform watershed detection (labels connected components with unique number
    ret, labels = cv2.connectedComponents(mask_mostreliable)
    # get indices which belong to each of the reliable clusters (will not be split by the margin)
    # bins_map [j-1] contains lin/col = yx = [0,1] indices of connected region j ([2, Npts])
    # take indices of nonzero (defined in line 267 below) so that the indices refer to the same mask
    region_idx_map = [(labels == j).nonzero() for j in range(1, np.max(labels)+1)]

    # TODO: use or remove
    # attribute point score?
    #points_SCORE = np.zeros([Ny, Nx], dtype=np.uint8)
    #points_SCORE += mask_in // 255  # score 1 to points in mask
    #points_SCORE += 10 * (cv2.morphologyEx(mask_in, cv2.MORPH_OPEN, kernel, iterations=1) // 255)
    #points_SCORE += 20 * (cv2.morphologyEx(mask_in, cv2.MORPH_OPEN, kernel, iterations=2) // 255)
    # ideas:
    #dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    #ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Take a "histogram" (x-profile) of the bottom half of the image
    histogram = np.sum(mask_input[mask_input.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    # Consider a margin to avoid locating lanes too close to border
    Ny, Nx = mask_input.shape[0:2]
    midpoint = np.int(Nx // 2)
    leftx_base = np.argmax(histogram[Nx//10:midpoint]) + Nx//10
    rightx_base = np.argmax(histogram[midpoint:Nx-Nx//10]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(Ny // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = mask_input.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base


    # Create empty lists to receive left and right lane pixel indices
    # inds ==> refer to the nonzero selection!
    left_lane_inds = []
    right_lane_inds = []
    # ind_regs ==> refer to the image, not nonzero (keep a separate list!)
    left_lane_inds_fromlabels = []
    right_lane_inds_fromlabels = []

    # Create an output image to draw on and visualize the result
    if NO_IMG == False:
        out_img = np.dstack((mask_input, mask_input, mask_input))
    else:
        out_img = None #no image returned

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

        # Append these indices to the lists (nonzero indices!)
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        #check the connected regions inside the selection
        #if points are found, add the whole region to the selection of good indices
        #kept a separate list as this refers to the image indices (and not nonzero, as the labeling requires the full image
        # and not only the non zero points of the mask)
        labels_in_left = np.unique(labels[nonzeroy[good_left_inds], nonzerox[good_left_inds]])
        labels_in_left = labels_in_left[labels_in_left > 0] #0 = background
        if np.size(labels_in_left) > 0:
            for k in labels_in_left:
                # y indices of the whole region ==> get portion of region within y-window
                yreg_left, xreg_left = region_idx_map[k-1][0],region_idx_map[k-1][1] #value of label k maps to index k-1!
                reg_good_idx = np.where((yreg_left >= win_y_low) & (yreg_left <= win_y_high))[0]
                #save 1D index and then convert back to xy later
                left_lane_inds_fromlabels.append(np.ravel_multi_index((yreg_left[reg_good_idx], xreg_left[reg_good_idx]),
                                                                       mask_input.shape))
                out_img[yreg_left[reg_good_idx], xreg_left[reg_good_idx]] = [0, 255, 255]
        else:
            xreg_left = []  # empty for concatenation


        #same for right
        labels_in_right = np.unique(labels[nonzeroy[good_right_inds], nonzerox[good_right_inds]])
        labels_in_right = labels_in_right[labels_in_right > 0] #0 = background
        if np.size(labels_in_right) > 0:
            for k in labels_in_right:
                #y indices of the whole region ==> get portion of region within y-window
                yreg_right, xreg_right = region_idx_map[k-1][0], region_idx_map[k-1][1]  #value of label k maps to index k-1!
                reg_good_idx = np.where((yreg_right >= win_y_low) & (yreg_right <= win_y_high))[0]
                #save 1D index and then convert back to xy later
                right_lane_inds_fromlabels.append(np.ravel_multi_index((yreg_right[reg_good_idx], xreg_right[reg_good_idx]),
                                                                       mask_input.shape))
                out_img[yreg_right[reg_good_idx], xreg_right[reg_good_idx]] = [255, 255, 0]
        else:
            xreg_right = []  # empty for concatenation

        # Update window's x center
        #left window
        x_detected_left = np.concatenate((nonzerox[good_left_inds],xreg_left))
        leftx_previous = leftx_current  # save current value
        # If > minpix found pixels, recenter next window #
        # (`right` or `leftx_current`) on their mean position #
        if np.size(x_detected_left) >= minpix:
            leftx_current = np.int64(np.mean(x_detected_left)) #update
        else: #not enough pixels in window, assume tendency of previous windows continues
            # make partial fit of order 2
            polycf_left = np.polyfit(nonzeroy[np.concatenate(left_lane_inds)],
                                     nonzerox[np.concatenate(left_lane_inds)], 2)
            leftx_current = np.int(np.round(np.polyval(polycf_left, 0.5*(win_y_low+win_y_high))))

        #right window
        x_detected_right = np.concatenate((nonzerox[good_right_inds],xreg_right))
        rightx_previous = rightx_current  # store last value before update
        if np.size(x_detected_right) >= minpix:
            rightx_current = np.int64(np.mean(x_detected_right))
        else:  # not enough pixels in window, assume tendency of previous window continues
            # make partial fit of order 2
            polycf_right = np.polyfit(nonzeroy[np.concatenate(right_lane_inds)],
                                     nonzerox[np.concatenate(right_lane_inds)], 2)
            rightx_current = np.int(np.round(np.polyval(polycf_right, 0.5*(win_y_low+win_y_high))))
        # store step w.r.t. last value for future iterations
    #for each window

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    # inds ==> refer to the nonzero selection!
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # append output from labels
    # inds_fromlabels ==> refer to the whole mask (use np.unravel --> y_idx, x_idx)
    # convert from list to array with np.concatenate
    if left_lane_inds_fromlabels != []:
        yidx, xidx = np.unravel_index(np.concatenate(left_lane_inds_fromlabels), mask_input.shape)
        leftx = np.concatenate((leftx, xidx))
        lefty = np.concatenate((lefty, yidx))
    #same for right lane
    if right_lane_inds_fromlabels != []:
        yidx, xidx = np.unravel_index(np.concatenate(right_lane_inds_fromlabels), mask_input.shape)
        # uniq sorts the inputs, so the indices are taken so the xy coordinates are kept together
        rightx = np.concatenate((rightx, xidx))
        righty = np.concatenate((righty, yidx))

    # Fit a second order polynomial to each using `np.polyfit`, assuming x = f(y) #
    polycf_left = np.polyfit(lefty, leftx, 2)
    polycf_right = np.polyfit(righty, rightx, 2)

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

        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
    # Optional Visualization Steps

    #wrap data into LaneLines objects
    leftlane = LaneLine(leftx, lefty, polycf_left)
    rightlane = LaneLine(rightx, righty, polycf_right)

    return leftlane, rightlane, lane_annotation, out_img
# ---------------------------------------------------------------------
def find_lane_xy_frompoly(mask_input, polycf_left, polycf_right, margin = 80, NO_IMG = False):
    """ Take the input mask and perform a search around the polynomial-matching area
     Return the coordinates of the located pixels, polynomial coefficients and optionally an image showing the
       windows/detections
     **Parameters/Keywords:
     margin ==> Set the width of the windows +/- margin
     NO_IMG ==> do not calculate the diagnose output image (used in pipeline) """

    #convert bool to byte
    mask_input = 255*np.uint8(mask_input/np.max(mask_input))
    Ny, Nx = np.shape(mask_input)

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

    # Fit new polynomials
    # Fit a second order polynomial to each with np.polyfit() : x = f(y)#
    polycf_left = np.polyfit(lefty, leftx, 2)
    polycf_right = np.polyfit(righty, rightx, 2)
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
    leftlane = LaneLine(leftx, lefty, polycf_left)
    rightlane = LaneLine(rightx, righty, polycf_right)

    return leftlane, rightlane, lane_annotation, out_img
# -------------------------------------------------------------------
def getlane_annotation(mask_shape, polycf_left, polycf_right, img2annotate=[], xmargin = 5, PLOT_LINES=False):
    """ Give the shape and the polynomial coefficients, return byte mask showing the region inside the two curves
        (plus an optional x-margin). Also add the annotations to an image if it is provided.
     **Parameters/Keywords:
     img2annotate ==> RGB image to annotate, if provided (should match the size of mask_shape!)
     xmargin ==> Set the width of the +/- margin "
     PLOT_LINES ==> Add the polynomial lines to output (only relevant if a img2annnotate is present)"""

    #get sizes and declare output
    Ny, Nx = mask_shape[0:2]
    out_mask = np.zeros([Ny, Nx], dtype=np.uint8)

    if np.size(img2annotate) != 0:
        if np.shape(img2annotate)[0:2] != (Ny, Nx): # note: must be tuple !
            raise Exception ("Error: Image to annotate must match mask size!")
        #blank RGB to add lane are to
        img_annotated = np.zeros_like(img2annotate)
    else:
        img_annotated = None
    #check if annotation is possible

    ploty = np.arange(Ny)  #y values to plot
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
        cv2.fillPoly(img_annotated, np.int_([lane_pts]), (0, 255, 0)) #green lane
        img_annotated = cv2.addWeighted(img2annotate, 1, img_annotated, 0.3, 0) #add to image
        # Plot the polynomial lines onto the image?
        if PLOT_LINES:
            plt.plot(left_fitx, ploty, color='yellow', linestyle='--')
            plt.plot(right_fitx, ploty, color='yellow', linestyle='--')
    #annotations

    return out_mask, img_annotated
# -------------------------------------------------------------------
