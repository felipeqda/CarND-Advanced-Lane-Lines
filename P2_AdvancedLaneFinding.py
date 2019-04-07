# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:42:57 2019

@author: Felipe
"""

INTERACTIVE_MODE = False #set to False to allow video processing

"""------------------------------------------------------------"""
"""imports"""
"""------------------------------------------------------------"""
# import packages
import matplotlib as mpl
if INTERACTIVE_MODE:
    mpl.use('Qt5Agg')  #view windows
else:
    # use non-interactive backend (print to "virtual window")
    mpl.use('Agg')

import matplotlib.pyplot as plt
##plt.rcParams.update({'backend': 'Qt5Agg'})
# modules
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import cv2
import os
from pdb import set_trace as stop  # useful for debugging
import glob  # to expand paths
from moviepy.editor import VideoFileClip

#import from subroutines
# image display
from P2_subroutines import plotNimg, weighted_img, restrict2ROI, closePolygon, get_plot, gaussian_blur
# calibration and perspective transformation
from P2_subroutines import calibrateCamera, Warp2TopDown
# thresholding/gradients
from P2_subroutines import abs_sobel_thresh, mag_thresh, dir_thresh, lanepxmask
# mask processing and fitting
from P2_subroutines import find_lane_xy_frompoly, find_lane_xy_frommask, getlane_annotation, cf_px2m, r_curve


"""------------------------------------------------------------"""
"""Step 1: Single-frame pipeline """
"""------------------------------------------------------------"""

#1) Single frame script for testing/development/debugging
def single_frame_analysis(input_image,
                          SHOW_COLOR_GRADIENT = True, SHOW_WARP = True,  SHOW_FIT = True,
                          sobel_kernel = 7):
    """  Single frame pipeline with detailed information, for development of steps and debugging of single frames
         **Parameters/Keywords
         # keywords to control level of output (set True for step by step analysis of image/frame)
         SHOW_COLOR_GRADIENT ==> show color channel analysis and gradients
         SHOW_WARP  ==> show warping
         SHOW_FIT   ==> show pixel detection and fitting step
         sobel_kernel ==> for gradient calculations """

    # take image name, make a directory for each, for output (used if plotting is on)
    if np.ndim(input_image) == 0: #string input
        img_basename = os.path.basename(input_image).split('.jpg')[0]
        output_dir = 'output_images' + os.sep + img_basename + os.sep
        if SHOW_COLOR_GRADIENT or SHOW_WARP or SHOW_FIT:
            os.makedirs(output_dir, exist_ok=True)
        # read image and get dimensions
        img_RGB = mpimg.imread(input_image)
        print('')
        print('Input Image file: ', img_basename, 'with dimensions:', img_RGB.shape)
        SAVE_PLOTS = True
    elif np.ndim(input_image) == 3:
        img_RGB = input_image
        SAVE_PLOTS = False
        print('')
        print('Input frame has dimensions:', img_RGB.shape)
    else:
        raise Exception('Error: Input string or image...')

    Ny, Nx, _ = np.shape(img_RGB)
    # print image info and plot
    plt.close('all')

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
        if SAVE_PLOTS: plt.savefig(output_dir + img_basename + '_grads.png')

        # check RGB channels
        # plot RGB channels for threshold analysis
        plotNimg([img_RGB, img_RGB[:, :, 0], img_RGB[:, :, 1], img_RGB[:, :, 2]],
                 ['RGB', 'R', 'G', 'B'],
                 [None, "gray", "gray", "gray"],
                 'Input image RGB color channels', fig_num=2)
        if SAVE_PLOTS: plt.savefig(output_dir + img_basename + '_RGB.png')

        # convert to HLS color space
        img_HLS = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HLS)
        # output ==> np.shape(img_HLS) = (Ny, Ny, 3 = [H, L, S])

        # plot HLS channels for threshold analysis
        plotNimg([img_RGB, img_HLS[:, :, 0], img_HLS[:, :, 1], img_HLS[:, :, 2]],
                 ['RGB', 'H', 'L', 'S'],
                 [None, "gray", "gray", "gray"],
                 'Input image HLS color channels', fig_num=3)
        if SAVE_PLOTS: plt.savefig(output_dir + img_basename + '_HLS.png')

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
    # get region of interest in image and warped domain (only relevant for plotting)
    x_ROI, y_ROI, pts_ROI = Perspective.xy_ROI_img()
    x_wROI, y_wROI = Perspective.xy_ROI_warped()

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
        plt.gca().set_xbound([0, Nx]) #keep x-axis within image
        plt.title('Input Image')
        plt.subplot(1, 2, 2)
        plt.imshow(Perspective.warp(img_RGB))
        plt.plot(x_warp, y_warp, 'r--')
        plt.plot(x_wROI, y_wROI, 'b')
        plt.title('Warped "top-down" Image')
        if SAVE_PLOTS: plt.savefig(output_dir + img_basename + '_warp.png')


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
        if SAVE_PLOTS: plt.savefig(output_dir + img_basename + '_masks-img.png')

    # B) warp before detect ==> less blurry
    #get warped RGB image
    imgRGB_warped = Perspective.warp(img_RGB)

    # color processing of warped image
    box_size = 20
    box_ystep = 30
    box_vertices = box_size * np.array([[(-1, -1),  # bottom left
                                         (-1, 1), (1, 1),
                                         (1, -1)]], dtype=np.int32)  # bottom right

    x_box, y_box = closePolygon(box_vertices)
    # get average color of boxes and mask out
    image_new = np.copy(imgRGB_warped)
    for i_box in range(3):
        box = imgRGB_warped[Ny-i_box*box_ystep:Ny-i_box*box_ystep-box_size,
                            Nx//2-box_size:Nx//2+box_size,:]
        avg_color = np.mean(box, axis=(0, 1))
        mask = cv2.inRange(gaussian_blur(imgRGB_warped, 9), np.uint8(avg_color - 50), np.uint8(avg_color + 50))
        image_new = cv2.bitwise_and(image_new, image_new, mask=255 - mask)
    imgRGB_warped = image_new #replace

    mask_warped, cbin_warped = lanepxmask(imgRGB_warped)


    plotNimg([imgRGB_warped, cbin_warped, mask_warped], ['RGB image (warped)', 'Mask components', 'Output mask'],
                 [None, None, "gray"], 'Warped Image and Detection mask', fig_num=7)
    # mark boxes
    plt.figure(num = 7)
    plt.subplot(2,2,1)
    for i_box in range(3):
        plt.plot(x_box+Nx//2,y_box-i_box*box_ystep,'k')
    if SAVE_PLOTS: plt.savefig(output_dir +'1'+ img_basename + '_masks-warp.png')


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
        if SAVE_PLOTS: plt.savefig(output_dir + '2' +img_basename + '_slidingwindowfit.png')

        # IDEA: fit from lane borders?
        # plt.figure(num=11)
        # gradx_mask = abs_sobel_thresh(imgRGB_warped, orient='x', thresh=(20, 100), sobel_kernel=sobel_kernel)
        # left, right, out_img = find_lane_xy_frommask(255*np.uint8(gradx_mask))
        # print('')
        # print('Alternative mask')
        # print('Left fit: ', left.cf)
        # print('Right fit: ', right.cf)
        # plt.imshow(out_img)
        # if SAVE_PLOTS: plt.savefig(output_dir + '20' +img_basename + '_slidingwindowfit2.png')

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
        if SAVE_PLOTS: plt.savefig(output_dir + '2' +img_basename + '_polycoeffit.png')


    """VI) Process fit and represent lane region in image """
    # force parallel but keep x_bottom (rather than the "c" coefficient!)
    # index: [left/right, a/b] ==> x = f(y) = a*y^2 + b*y + c
    cfs = np.vstack((left.cf, right.cf))
    cf_variance = np.vstack((left.C_matrix[[0,1],[0,1]], right.C_matrix[[0,1],[0,1]]))
    #average a/b coefficients with inverse-variance weights
    w =  np.sum(cf_variance, axis=0)/cf_variance
    cf_avg = np.mean(w*cfs[:,0:2], axis=0)/np.mean(w, axis=0)
    # update coefficients, forcing parallel but keeping x_bottom = f(Ny)
    left.cf[0:2] = cf_avg
    left.cf[2] = left.x_bottom - np.polyval(np.concatenate((cf_avg, [0])),  Ny)
    right.cf[0:2] = cf_avg
    right.cf[2] = right.x_bottom - np.polyval(np.concatenate((cf_avg, [0])),  Ny)
    print('')
    print('Final results for display')
    print('Left fit: ', left.cf)
    print('Right fit: ', right.cf)

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

    img_out = cv2.addWeighted(img_RGB, 1, lane_annotation, 0.3, 0)  # add to RGB image
    imglane = Perspective.unwarp(imglane_warped)

    if SHOW_FIT:
        plotNimg([fit_img, imglane_warped, imglane, img_out],
                 ['Fit of lanes', 'Lane area (warped)', 'Lane area (unwarped)', 'Output Image'],
                 [None, None, None, None], 'Detected Lane Area', fig_num=14)
        #mark a region of interest matching the detection area
        #xROI, yROI = Perspective.xy_ROI_img()
        #plt.plot(xROI, yROI, 'b--', linewidth=2)
        if SAVE_PLOTS: plt.savefig(output_dir +'3'+ img_basename + '_lane.png')


    """VI) Convert from pixel coordinates to m and compute R_curvature/distance from center """

    # convert coefficients to m
    cf_meters_left = cf_px2m(left.cf, img_RGB.shape)
    cf_meters_right = cf_px2m(right.cf, img_RGB.shape)
    # calculate curvature, center of lane in px and deviation from center of image/car center in m
    curvature = r_curve(cf_meters_left, 0.0) #bottom ==> y = 0.0 in new coordinate system
    sign_Curv = np.sign(cf_meters_left[0])  # positive to the right
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

    #plots an arrow from xytext to xy (ending near point)
    plt.annotate("", xy=(x_center_px+sign_Curv*inv_Curv*(200), 3*Ny/4), xytext=(x_center_px, 3*Ny/4),
                 arrowprops=dict(arrowstyle='simple', facecolor = my_cmp(1.0-inv_Curv)),
                 xycoords = 'data', fontsize=12)
    # print current lane center distance from camera center in m
    plt.text(Nx/2, 85, "$\Delta x_{center}$"+" = {:.2f} [m]".format(deltax_center),
             horizontalalignment='left',verticalalignment='center',
             fontsize=12, weight='normal', color='w')
    # save to both directories
    if SAVE_PLOTS:
        plt.savefig(output_dir +'4'+ img_basename + '_output.png')
        plt.savefig('output_images' + os.sep + img_basename + '_output.jpg', format='jpg')
"""------------------------------------------------------------"""

# image_list = os.listdir('./test_images')
image_list = ['straight_lines1.jpg',
              'straight_lines2.jpg',
              'test1.jpg',
              'test2.jpg',  #left curve
              'test3.jpg',
              'test4.jpg',
              'test5.jpg',
              'test6.jpg']

#### Run for single input
#input_image = "test_images" + os.sep + image_list[3] #5!  #indices [0-7]
#single_frame_analysis(input_image)

#### Run for all
#for input_image in image_list: single_frame_analysis("test_images" + os.sep + input_image)


#2) Pipeline for video processing (use class to store cross-frame information)

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
        self.cfs_uncertainty_buffer = np.zeros([N_buffer, 3], dtype='float')
        self.cfs_uncertainty_buffer[:] = np.nan  # invalid values
        self.cfs_uncertainty_avg = np.zeros([3], dtype='float')
        self.cfs_uncertainty_std = np.zeros([3], dtype='float')


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
    def update_stats(self, new_cfs, new_C_matrix):
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

        self.cfs_uncertainty_buffer = np.roll(self.cfs_uncertainty_buffer, 1, axis=0)
        new_cf_uncertainty = new_C_matrix[[0, 1, 2], [0, 1, 2]] #diagonal of covariance matrix (variances)
        self.cfs_uncertainty_buffer[0, :] = new_cf_uncertainty
        self.cfs_uncertainty_avg = np.nanmean(self.cfs_uncertainty_buffer, axis=0)
        self.cfs_uncertainty_std = np.nanstd(self.cfs_uncertainty_buffer, axis=0)

        self.cfs_current = new_cfs
    # -------------------------------
    def update_xy(self, x_new, y_new):
        # update list of x values
        if len(self.recent_xfitted) == self.N_buffer: #full buffer
            oldest = self.recent_xfitted.pop(0) #this removes this entry
        self.recent_xfitted.append(x_new)
        # bookkeeping operations
        self.allx = x_new
        self.ally = y_new
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
class ProcessFrame:
    """ Main pipeline as a class to store line info/stats across frames"""
    # -------------------------------
    def __init__(self, N_buffer = 10, sobel_kernel_sz = 7):
        self.N_frames = 0
        self.N_buffer = N_buffer
        # objects of LaneLine4Video class
        self.leftlane = LaneLine4Video(N_buffer = N_buffer)
        self.rightlane = LaneLine4Video(N_buffer = N_buffer)
        # set parameters for pipeline (expand list for increased generality)
        self.sobel_ksize = sobel_kernel_sz

        """ set tables and constants which do not need to be performed more than once """
        # store Calibration table
        self.cal_mtx, self.dist_coef = calibrateCamera(FORCE_REDO=False)
        # store perspective transformation object
        self.Perspective = Warp2TopDown()

        # store curvature ranges and color map used to highligh degree of curvature
        # http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm#BGBHGEGC
        self.min_rcurve = np.min(0.3048*np.array([643, 4575])) #absolute minimum values
        # create a colormap which is red-yellow-green in the ranges
        # provide a dictionary with 'red', 'green', 'blue' keys with (x, y0, y1) tuples
        # x[i] < x_in < x[i+1] => color between y1[i] and y0[i+1]
        # x = 0.0 - 0.5 - 1.0 ==> R, Y='FFFF00', G
        cdict_anchor = {'red':   [[0.0,  1.0, 1.0], [0.5, 1.0, 1.0], [1.0, 0.0, 0.0]],
                        'green': [[0.0,  0.0, 0.0], [0.5, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        'blue':  [[0.0,  0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]}
        self.my_cmp = LinearSegmentedColormap('GoodBad', segmentdata=cdict_anchor, N=256)

    # -------------------------------
    def __call__(self, frame):
        """  Pipeline for lane annotation in frame  """
        sobel_kernel = self.sobel_ksize #for gradient calculations

        # get dimensions
        Ny, Nx, _ = np.shape(frame)

        """I) apply calibration"""
        frame = cv2.undistort(frame, self.cal_mtx, self.dist_coef, None, self.cal_mtx)

        """II) Get warped detection masks"""
        # warp before detect ==> less blurry
        # get warped RGB frame
        frame_warped = self.Perspective.warp(frame)
        mask_warped, _ = lanepxmask(frame_warped)

        """III) Process mask and get polynomial fit for boundaries """
        #### HERE
        if (self.N_frames < self.N_buffer): #starting up
            # fit from the mask (no a-priori knowledge)
            # returns LaneLine objects
            left_currentframe, right_currentframe, left_right_pxs, _ = find_lane_xy_frommask(mask_warped, NO_IMG=True)
            self.leftlane.detected = True
            self.rightlane.detected = True
        else: #try to use a-priori knowledge
            # fit from a-priori knowledge (method M1)
            cf_left, cf_right = self.leftlane.cfs_median, self.rightlane.cfs_median
            # returns LaneLine objects
            left_currentframe, right_currentframe, left_right_pxs, _ = find_lane_xy_frompoly(mask_warped,
                                                                                            cf_left, cf_right, NO_IMG=True)
            # evaluate uncertainty of coefficients
            # a coefficient is considered bad if the variance is above average + a threshold
            BAD_LEFT_M1 = np.any(left_currentframe.C_matrix[[0,1,2],[0,1,2]] >
                              (self.leftlane.cfs_uncertainty_avg + 2*self.leftlane.cfs_uncertainty_std))
            BAD_RIGHT_M1 = np.any(right_currentframe.C_matrix[[0,1,2],[0,1,2]] >
                              (self.rightlane.cfs_uncertainty_avg + 2*self.rightlane.cfs_uncertainty_std))
            # if any of those is bad, try the other method (M2)
            if BAD_LEFT_M1 or BAD_RIGHT_M1:
                # detect coefficients based on mask (has to be done for both lanes)
                left_currentframe2, right_currentframe2, _, _ = find_lane_xy_frommask(mask_warped, NO_IMG=True)
                BAD_LEFT_M2 = np.any(left_currentframe2.C_matrix[[0, 1, 2], [0, 1, 2]] >
                                    (self.leftlane.cfs_uncertainty_avg + 2*self.leftlane.cfs_uncertainty_std))
                BAD_RIGHT_M2 = np.any(right_currentframe2.C_matrix[[0, 1, 2], [0, 1, 2]] >
                                     (self.rightlane.cfs_uncertainty_avg + 2*self.rightlane.cfs_uncertainty_std))
                # take more reliable estimate, or median of buffer
                if BAD_LEFT_M1 and not(BAD_LEFT_M2): #second estimate is better, replace
                    left_currentframe = left_currentframe2
                elif BAD_LEFT_M1 and BAD_LEFT_M2: #both are bad, take median
                    left_currentframe.cfs = self.leftlane.cfs_median
                    # set variance levels to average, these values will influence weights later
                    left_currentframe.C_matrix[[0,1],[0,1]]= self.leftlane.cfs_uncertainty_avg[0:2]
                if BAD_RIGHT_M1 and not(BAD_RIGHT_M2): #second estimate is better, replace
                    right_currentframe = right_currentframe2
                elif BAD_RIGHT_M1 and BAD_RIGHT_M2:  # both are bad, take median
                    right_currentframe.cfs = self.rightlane.cfs_median
                    # set variance levels to average, these values will influence weights later
                    left_currentframe.C_matrix[[0,1],[0,1]] = self.leftlane.cfs_uncertainty_avg[0:2]
                # take most reliable or median
        # buffer is available

        # update buffer with information from each lane (info from frame, no smoothing)
        self.N_frames= self.N_frames+1
        # left lane
        self.leftlane.update_stats(left_currentframe.cf, left_currentframe.C_matrix)
        self.leftlane.update_xy(left_currentframe.x_pix, left_currentframe.y_pix)
        # right lane
        self.rightlane.update_stats(right_currentframe.cf, right_currentframe.C_matrix)
        self.rightlane.update_xy(right_currentframe.x_pix, right_currentframe.y_pix)

        # smooth coefficients over buffer for display
        left_currentframe.cf, right_currentframe.cf = self.leftlane.cfs_median, self.rightlane.cfs_median
        # use average uncertainties to model effect of smoothing filter
        left_currentframe.C_matrix[[0, 1], [0, 1]] = self.leftlane.cfs_uncertainty_avg[0:2]
        right_currentframe.C_matrix[[0, 1], [0, 1]] = self.rightlane.cfs_uncertainty_avg[0:2]

        """IV) Process fit and represent lane region in image """
        # force parallel ==> in-frame consistency
        # index: [left/right, a/b] ==> x = f(y) = a*y^2 + b*y + c
        cfs = np.vstack((left_currentframe.cf, right_currentframe.cf))
        cf_variance = np.vstack((left_currentframe.C_matrix[[0,1],[0,1]], right_currentframe.C_matrix[[0,1],[0,1]]))
        #average a/b coefficients with inverse-variance weights
        w =  np.sum(cf_variance, axis=0)/cf_variance
        cf_avg = np.mean(w*cfs[:,0:2], axis=0)/np.mean(w, axis=0)
        # update coefficients, forcing parallel but keeping x_bottom = f(Ny)
        left_currentframe.cf[0:2] = cf_avg
        left_currentframe.cf[2] = left_currentframe.x_bottom - np.polyval(np.concatenate((cf_avg, [0])), Ny)
        right_currentframe.cf[0:2] = cf_avg
        right_currentframe.cf[2] = right_currentframe.x_bottom - np.polyval(np.concatenate((cf_avg, [0])), Ny)


        """ V) Annotate frame with lane markings """
        # warp back annotation
        lane_mask_warped, _ = getlane_annotation(mask_warped.shape,
                                                 left_currentframe.cf, right_currentframe.cf,
                                                 img2annotate=[],
                                                 xmargin = 0, PLOT_LINES=False)
        # post-processing (necessary not to lose ROI)
        lane_mask = self.Perspective.unwarp(lane_mask_warped)
        left_right_annotation = self.Perspective.unwarp(left_right_pxs)
        lane_annotation = np.dstack((left_right_annotation[:,:,0],
                                     lane_mask,
                                     left_right_annotation[:,:,2])) #auxiliar to mark lane in green

        frame_out = cv2.addWeighted(frame, 1, lane_annotation, 0.3, 0)  # add to RGB frame


        """VI) Convert from pixel coordinates to m and compute R_curvature/distance from center """
        # convert coefficients to m
        cf_meters_left = cf_px2m(left_currentframe.cf, frame.shape)
        cf_meters_right = cf_px2m(right_currentframe.cf, frame.shape)
        # calculate curvature, center of lane in px and deviation from center of image/car center in m
        curvature = r_curve(cf_meters_left, 0.0) #bottom ==> y = 0.0 in new coordinate system
        sign_Curv = np.sign(cf_meters_left[0]) #positive to the right
        x_center_px = 0.5*(left_currentframe.x_bottom+right_currentframe.x_bottom)
        deltax_center = 0.5*(cf_meters_left[2]+cf_meters_right[2])
        # update classes
        self.leftlane.update_R_coords(curvature, deltax_center)
        self.rightlane.update_R_coords(curvature, deltax_center)
        # get smoothened values from objects (left and right are forced to be consistent)
        curvature, deltax_center = self.leftlane.radius_of_curvature, self.leftlane.line_center_offset


        # use inverse [-->0 for straight line, normalize to 1 for minimum possible values] curvature to scale color
        inv_Curv = (1/curvature)/(1/self.min_rcurve) #[0-1]

        # print current curvature to output frame
        fig_aux = plt.figure(figsize=(Nx/100.0, Ny/100.0), num=100)  #does not result in an open window in the "Agg" backend
        plt.clf()
        fig_aux.add_axes([0,0,1,1]) #use whole area
        plt.imshow(frame_out)
        plt.text(Nx/2, 40, r"$R_{curve} = $",#+"{:.2f} [m]".format(curvature),
                 horizontalalignment='left',verticalalignment='center',
                 fontsize=14, weight='bold', color='w')
        plt.text(Nx/2+80, 40, "{:.2f} [m]".format(curvature),
                 horizontalalignment='left',verticalalignment='center',
                 fontsize=14, weight='bold', color=self.my_cmp(1.0-inv_Curv))
        #plots an arrow from xytext to xy (ending near text)
        plt.annotate("", xy=(x_center_px+sign_Curv*inv_Curv*(200), 3*Ny/4), xytext=(x_center_px, 3*Ny/4),
                     arrowprops=dict(arrowstyle='simple', facecolor = self.my_cmp(1.0-inv_Curv)),
                     xycoords = 'data', fontsize=14)
        # print current lane center distance from camera center in m
        plt.text(Nx/2, 85, "$\Delta x_{center}$"+" = {:.2f} [m]".format(deltax_center),
                 horizontalalignment='left',verticalalignment='center',
                 fontsize=14, weight='normal', color='w')
        plt.axis("off")
        frame_out[:, :, :] = get_plot(fignum=100)
        return frame_out
# ---------------------------------------------------------------------



# TODO: test with video

test_videos = ['project_video.mp4', 'challenge_video.mp4', 'harder_challenge_video.mp4']
input_video = test_videos[1] # choose video
output_video = os.path.basename(input_video).split('.mp4')[0]+'_output.mp4'


# applies process_image frame by frame
processing_pipeline = ProcessFrame(N_buffer = 10)
print('')
print('Input video: '+input_video)
QUICK_TEST = False
if QUICK_TEST:
    #take a short sub-clip for the testing
    clip_in = VideoFileClip(input_video).subclip(0, 5)
    print('Processing only an excerpt...')
else:
    clip_in = VideoFileClip(input_video)

# show a frame based on the time tag
def debug_frame(t):
    #fig = plt.figure()
    #fig.canvas.set_window_title('Debug t ='+str(t))
    plt.imshow(processing_pipeline(clip_in.get_frame(t)))
#debug_frame(0.0)

# process video
PROCESS_VIDEO = True
if PROCESS_VIDEO:
    clip_out = clip_in.fl_image(processing_pipeline)
    clip_out.write_videofile(output_video, audio=False)


#single_frame_analysis(clip_in.get_frame(0.0), SHOW_COLOR_GRADIENT = False, SHOW_WARP = False,  SHOW_FIT = True)


# %matplotlib qt5

#single_frame_analysis(clip_in.get_frame(40.081), SHOW_COLOR_GRADIENT = False, SHOW_WARP = False,  SHOW_FIT = True)
#processing_pipeline(clip_in.get_frame(2.36))