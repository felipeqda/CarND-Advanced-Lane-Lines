# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:42:57 2019

@author: Felipe
"""

"""------------------------------------------------------------"""
"""imports"""
"""------------------------------------------------------------"""
# import packages

import matplotlib.pyplot as plt
plt.rcParams.update({'backend': 'Qt5Agg'})
# modules
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import cv2
import os
from pdb import set_trace as stop  # useful for debugging
import glob  # to expand paths

#import from subroutines
# image display
from P2_subroutines import plotNimg, weighted_img, restrict2ROI, closePolygon
# calibration and perspective transformation
from P2_subroutines import calibrateCamera, Warp2TopDown
# thresholding/gradients
from P2_subroutines import abs_sobel_thresh, mag_thresh, dir_thresh, lanepxmask
# mask processing and fitting
from P2_subroutines import find_lane_xy_frompoly, find_lane_xy_frommask, getlane_annotation, cf_px2m, r_curve


"""------------------------------------------------------------"""
"""Step 1: Single-frame pipeline """
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

#### INPUTIMAGE
input_image = "test_images" + os.sep + image_list[7] #5!  #indices [0-7]

#def single_frame_analysis(input_image):
#    """  Single frame pipeline with detailed information, for development of steps and debugging of single frames """
# keywords to control level of output (set True for step by step analysis of image/frame)
SHOW_COLOR_GRADIENT = False  # show color channel analysis and gradients
SHOW_WARP = False            # show warping
SHOW_FIT = True              # show pixel detection and fitting step
sobel_kernel = 7             # for gradient calculations

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
cal_mtx, dist_coef = calibrateCamera(FORCE_REDO=False)
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


    # # "negative" threshold to reject artifacts?
    # roadside_mask = (img_HLS[:,:,0] < 40) & (img_HLS[:,:,2] > 30)  & (img_HLS[:,:,2] < 120)
    # asphalt_mask = (img_HLS[:,:,0] > 100) & (img_HLS[:,:,2] < 30)
    # #roadside_mask = (img_HLS[:,:,2] <= S_thd[0])
    # reject_mask = (~roadside_mask) & (~asphalt_mask)
#plot detailed analysis (useful to verify properties and select thresholds)



"""III) Perspective Transformations and Region of Interest (ROI) """
Perspective = Warp2TopDown() #object to perform warping

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


plotNimg([imgRGB_warped, cbin_warped, mask_warped], ['RGB image (warped)', 'Mask components', 'Output mask'],
             [None, None, "gray"], 'Warped Image and Detection mask', fig_num=7)
plt.savefig(output_dir +'1'+ img_basename + '_masks-warp.png')


"""V) Process mask and get polynomial fit for boundaries """

#fit from the mask (no a-priori knowledge)
left, right, left_right_pxs, fit_img = find_lane_xy_frommask(mask_warped)
# take output of previous step to test a-posteriori fitting
cf_1, cf_2 = left.cf, right.cf
left2, right2, _, fit_img2 = find_lane_xy_frompoly(mask_warped, cf_1, cf_2)

if SHOW_FIT:
    fig = plt.figure(num=10)
    fig.canvas.set_window_title("Detected lanes: from pixel mask")
    print('')
    print('Fit from pixel search')
    print('Left fit: ', left.cf)
    print('Right fit: ', right.cf)
    plt.imshow(fit_img)
    plt.savefig(output_dir + '2' +img_basename + '_slidingwindowfit.png')

    # IDEA: fit from lane borders?
    # plt.figure(num=11)
    # gradx_mask = abs_sobel_thresh(imgRGB_warped, orient='x', thresh=(20, 100), sobel_kernel=sobel_kernel)
    # left, right, out_img = find_lane_xy_frommask(255*np.uint8(gradx_mask))
    # print('')
    # print('Alternative mask')
    # print('Left fit: ', left.cf)
    # print('Right fit: ', right.cf)
    # plt.imshow(out_img)
    # plt.savefig(output_dir + '20' +img_basename + '_slidingwindowfit2.png')

    # Input coefficients for lane search (x = f(y))
    # FORCE
    #cf_1 =  [ 6.39534621e-04, -8.23499426e-01,  6.47270127e+02] #left
    #cf_2 =  [ 4.29743758e-04, -6.81291447e-01,  1.24452944e+03] #right

    fig = plt.figure(num=13)
    fig.canvas.set_window_title("Detected lanes: from polynomial coefficients")
    print('')
    print('Fit from polynomial coefficients')
    print('Left fit: ', left2.cf)
    print('Right fit: ', right2.cf)
    plt.imshow(fit_img2)
    plt.savefig(output_dir + '2' +img_basename + '_polycoeffit.png')


"""VI) Process fit and represent lane region in image """
# TODO: smoothing/harmonization/confidence weighting
# force parallel
# index: [left/right, a/b] ==> x = f(y) = a*y^2 + b*y + c
cfs = np.vstack((left.cf, right.cf))
cf_variance = np.vstack((left.C_matrix[[0,1],[0,1]], right.C_matrix[[0,1],[0,1]]))
#average a/b coefficients with inverse-variance weights
w =  np.sum(cf_variance, axis=0)/cf_variance
cf_avg = np.mean(w*cfs[:,0:2], axis=0)/np.mean(w, axis=0)
left.cf[0:2] = cf_avg
right.cf[0:2] = cf_avg

# warp back annotation
lane_mask_warped, imglane_warped = getlane_annotation(mask_warped.shape,
                                                      left.cf, right.cf,
                                                      img2annotate=imgRGB_warped,
                                                      xmargin = 0, PLOT_LINES=True)

# post-processing (necessary not to lose ROI)
lane_mask = Perspective.unwarp(lane_mask_warped)
left_right_annotation = Perspective.unwarp(left_right_pxs)
lane_annotation = np.dstack((left_right_annotation[:,:,0],
                             lane_mask,
                             left_right_annotation[:,:,2])) #auxiliar to mark lane in green

img_out = cv2.addWeighted(img_RGB, 1, lane_annotation, 0.5, 0)  # add to RGB image
imglane = Perspective.unwarp(imglane_warped)

if SHOW_FIT:
    plotNimg([fit_img, imglane_warped, imglane, img_out],
             ['Fit of lanes', 'Lane area (warped)', 'Lane area (unwarped)', 'Output Image'],
             [None, None, None, None], 'Detected Lane Area', fig_num=14)
    #mark a region of interest matching the detection area
    #xROI, yROI = Perspective.xy_ROI_img()
    #plt.plot(xROI, yROI, 'b--', linewidth=2)
    plt.savefig(output_dir +'3'+ img_basename + '_lane.png')


"""VI) Convert from pixel coordinates to m and compute R_curvature/distance from center """

# convert coefficients to m
cf_meters_left = cf_px2m(left.cf, img_RGB.shape)
cf_meters_right = cf_px2m(right.cf, img_RGB.shape)
# calculate curvature, center of lane in px and deviation from center of image/car center in m
curvature = r_curve(cf_meters_left, 0.0) #bottom ==> y = 0.0 in new coordinate system
x_center_px = 0.5*(left.x_bottom+right.x_bottom)
deltax_center = 0.5*(cf_meters_left[2]+cf_meters_right[2])


# curvature ranges (used to highligh color of curvature)
# http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm#BGBHGEGC
curv_rgs = 0.3048*np.array([643, 4575]) #absolute minimum values
# create a colormap which is red-yellow-green in the ranges
# provide a dictionary with 'red', 'green', 'blue' keys with (x, y0, y1) tuples
# x[i] < x_in < x[i+1] => color between y1[i] and y0[i+1]
# x = 0.0 - 0.5 - 1.0 ==> R, Y='FFFF00', G
cdict_anchor = {'red':   [[0.0,  1.0, 1.0], [0.5, 1.0, 1.0], [1.0, 0.0, 0.0]],
                'green': [[0.0,  0.0, 0.0], [0.5, 1.0, 1.0], [1.0, 1.0, 1.0]],
                'blue':  [[0.0,  0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]}
my_cmp = LinearSegmentedColormap('GoodBad', segmentdata=cdict_anchor, N=256)
# use inverse [-->0 for straight line, normalize to 1 for minimum possible values] curvature to scale color
inv_Curv = (1/curvature)/(1/np.min(curv_rgs)) #[0-1]

# plot final output
fig = plt.figure(num=15)
fig.canvas.set_window_title("Output frame")
plt.imshow(img_out)

# print current curvature
plt.text(Nx/2, 40, r"$R_{curve} = $",#+"{:.2f} [m]".format(curvature),
         horizontalalignment='left',verticalalignment='center',
         fontsize=12, weight='bold', color='w')
plt.text(Nx/2+180, 40, "{:.2f} [m]".format(curvature),
         horizontalalignment='left',verticalalignment='center',
         fontsize=12, weight='bold', color=my_cmp(1.0-inv_Curv))

#plots an arrow from xytext to xy (ending near text)
plt.annotate("", xy=(x_center_px+inv_Curv*(200), 3*Ny/4), xytext=(x_center_px, 3*Ny/4),
             arrowprops=dict(arrowstyle='simple', facecolor = my_cmp(1.0-inv_Curv)),
             xycoords = 'data', fontsize=12)
# print current lane center distance from camera center in m
plt.text(Nx/2, 85, "$\Delta x_{center}$"+" = {:.2f} [m]".format(deltax_center),
         horizontalalignment='left',verticalalignment='center',
         fontsize=12, weight='normal', color='w')
plt.savefig(output_dir +'4'+ img_basename + '_output.png')



# TODO: test with video
# TODO: convert px to m
