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
import cv2
from pdb import set_trace as stop  # useful for debugging

"""------------------------------------------------------------"""
""" Image display                                              """
"""------------------------------------------------------------"""

# ---------------------------------------------------------------------
def weighted_img(img_annotation, initial_img, α=0.8, β=1., γ=0.):
    ###mixes two color images of the same shape, img and initial_img, for annotations
    return cv2.addWeighted(initial_img, α, img_annotation, β, γ)


# ---------------------------------------------------------------------
def gaussian_blur(img, kernel_size):
    ###Applies a Gaussian Noise kernel###
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


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

    # TODO: save!


# ---------------------------------------------------------------------

"""------------------------------------------------------------"""
""" Masking and Thresholding Image or Gradient                 """
"""------------------------------------------------------------"""

# Threshold application, as developed in quizzes
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


"""------------------------------------------------------------"""
""" Miscellaneous                                              """
"""------------------------------------------------------------"""

def closePolygon(pts_xy):
    # get a poligon with [N, 2] vertices and returns x and y for plotting
    # by repeating the last point
    x = np.concatenate((pts_xy[:, 0], [pts_xy[0, 0]]))
    y = np.concatenate((pts_xy[:, 1], [pts_xy[0, 1]]))
    return x, y



"""------------------------------------------------------------"""
""" Mask processing and fitting                                """
"""------------------------------------------------------------"""
#functions to process the mask by finding lane pixels and fitting
#(based on quizzes)


# ---------------------------------------------------------------------
class LaneLine:
    #store the coordinates of the pixels and the polynomial coefficients for each lane
    #also store where the line reaches the bottom of the image
    def __init__(self, x_coord, y_coord, poly_coef):
        self.x_pix = x_coord
        self.y_pix = y_coord
        self.cf = poly_coef
        self.x_bottom = np.polyval(poly_coef, np.max(y_coord))
# ---------------------------------------------------------------------



# ---------------------------------------------------------------------
def find_lane_xy_frommask(mask_input, nwindows = 9, margin = 100, minpix = 50, NO_IMG = False):
    """ Take the input mask and perform a sliding window search
     Return the coordinates of the located pixels, polynomial coefficients and optionally an image showing the
       windows/detections
     Parameters:
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
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(mask_input.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = mask_input.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # keep track of the step in the x center of the last windows to improve tracking
    # (if no pixels are found, the tendency is assumed to continue, to avoid stacking of windows)
    xstep_lastwin = [0, 0]  #[left, right]

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
        win_y_low = mask_input.shape[0] - (window + 1) * window_height
        win_y_high = mask_input.shape[0] - window * window_height
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

        #check the connected regions inside the selection
        #if points are found, add the whole region to the selection of good indices
        #kept a separate list as this refers to the image indices (and not nonzero, as the labeling requires the full image
        # and not only the non zero points of the mask)
        labels_in_left = np.unique(labels[nonzeroy[good_left_inds], nonzerox[good_left_inds]])
        labels_in_left = labels_in_left[labels_in_left > 0] #0 = background
        if labels_in_left != []:
            for k in labels_in_left:
                #y indices of the whole region ==> get portion of region within y-window
                yreg = region_idx_map[k-1][0] #value of label k maps to index k-1!
                reg_good_idx = np.where((yreg >= win_y_low) & (yreg <= win_y_high))[0]
                left_lane_inds_fromlabels.append(reg_good_idx)
        #same for right
        labels_in_right = np.unique(labels[nonzeroy[good_right_inds], nonzerox[good_right_inds]])
        labels_in_right = labels_in_right[labels_in_right > 0] #0 = background
        if window == 8: stop()
        if labels_in_right != []:
            for k in labels_in_right:
                #y indices of the whole region ==> get portion of region within y-window
                yreg = region_idx_map[k-1][0]  #value of label k maps to index k-1!
                reg_good_idx = np.where((yreg >= win_y_low) & (yreg <= win_y_high))[0]
                right_lane_inds_fromlabels.append(reg_good_idx)


        # Append these indices to the lists (should be full-image indices!)
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Update window x center
        #left window
        leftx_previous = leftx_current  # save value
        # If > minpix found pixels, recenter next window #
        # (`right` or `leftx_current`) on their mean position #
        if np.size(good_left_inds) >= minpix:
            leftx_current = np.int64(np.mean(nonzerox[good_left_inds])) #update
        else: #not enough pixels in window, assume tendency of previous window continues
            leftx_current = leftx_previous + xstep_lastwin[0] #update with the same step
        # store step w.r.t. last value for future iterations
        xstep_lastwin[0] = leftx_current-leftx_previous
        #right window
        rightx_previous = rightx_current  # store last value before update
        if np.size(good_right_inds) >= minpix:
            rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))
        else:  # not enough pixels in window, assume tendency of previous window continues
            rightx_current = rightx_previous + xstep_lastwin[1] #update with the same step
        # store step w.r.t. last value for future iterations
        xstep_lastwin[1] = rightx_current - rightx_previous

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
    if right_lane_inds_fromlabels != []:
        yidx, xidx = np.unravel_index(np.concatenate(right_lane_inds_fromlabels), mask_input.shape)
        # uniq sorts the inputs, so the indices are taken so the xy coordinates are kept together
        rightx = np.concatenate((rightx, xidx))
        righty = np.concatenate((righty, yidx))

    # Fit a second order polynomial to each using `np.polyfit`, assuming x = f(y) #
    polycf_left = np.polyfit(lefty, leftx, 2)
    polycf_right = np.polyfit(righty, rightx, 2)

    # Optional Visualization Steps
    if NO_IMG == False:
        # Set colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Generate x and y values for plotting
        ploty = np.linspace(0, mask_input.shape[0] - 1, mask_input.shape[0])

        left_fitx = np.polyval(polycf_left, ploty)
        right_fitx = np.polyval(polycf_right, ploty)

        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
    # Optional Visualization Steps

    #wrap data into LaneLines objects
    leftlane = LaneLine(leftx, lefty, polycf_left)
    rightlane = LaneLine(rightx, righty, polycf_right)

    return  leftlane, rightlane, out_img
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
def find_lane_xy_frompoly(mask_input, polycf_left, polycf_right, margin = 100, NO_IMG = False):
    # Take the input mask and perform a search around the polynomial-matching area
    # Return the coordinates of the located pixels, polynomial coefficients and optionally an image showing the
    #   windows/detections
    # Parameters:
    # margin ==> Set the width of the windows +/- margin
    # NO_IMG ==> do not calculate the diagnose output image (used in pipeline)

    #convert to byte
    mask_input = 255*np.uint8(mask_input/np.max(mask_input))


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
    #  Fit a second order polynomial to each with np.polyfit() : x = f(y)#
    polycf_left = np.polyfit(lefty, leftx, 2)
    polycf_right = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, mask_input.shape[0] - 1, mask_input.shape[0])
    # Evaluate both polynomials using ploty, polycf_left and polycf_right #
    left_fitx = np.polyval(polycf_left, ploty)
    right_fitx = np.polyval(polycf_right, ploty)

    # Optional Visualization Steps
    if NO_IMG == False:
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((mask_input, mask_input, mask_input)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

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

    #wrap data into LaneLines objects
    leftlane = LaneLine(leftx, lefty, polycf_left)
    rightlane = LaneLine(rightx, righty, polycf_right)

    return  leftlane, rightlane, out_img
#-------------------------------------------------------------------