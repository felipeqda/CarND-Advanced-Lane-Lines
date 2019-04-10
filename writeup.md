# Advanced Lane Finding Project
#### Writeup: Felipe Queiroz de Almeida


****
### Project Description and Goals

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)


[image1]: ./output_images/calibration/calibration1.jpg "Calibration Input"
[image2]: ./output_images/calibration/calibration1_output.jpg "Calibration Output"
[image3]: ./output_images/test1/test1_calibration.png "Calibration of Frames"
[image4]: ./output_images/straight_lines1/straight_lines1_masks-img.png "Detection Masks"
[image5]: ./output_images/straight_lines1/straight_lines1_warp.png "Warping Example"
[image6]: ./output_images/straight_lines1/1straight_lines1_masks-warp.png "Detection Masks AFTER Warping"
[image7]: ./output_images/test2/test2_detailonwindowsearch.png "Sliding Window Search: Window Center Prediction"
[image8]: ./output_images/videoframes&misc/challengevideo1_detailonwindowsearch.png "Sliding Window Search: Window Center Prediction (steep curve)"
[image9]: ./output_images/test3/2test3_slidingwindowfit.png "Sliding Window Search: Result 1"
[image10]: ./output_images/test4/2test4_slidingwindowfit.png "Sliding Window Search: Result 2"
[image11]: ./output_images/test3/2test3_polycoeffit.png "Coefficient-Based Search: Result 1"
[image12]: ./output_images/test4/2test4_polycoeffit.png "Coefficient-Based Search: Result 2"
[image13]: ./output_images/test2_output.jpg "Output Frame 1"
[image14]: ./output_images/test3_output.jpg "Output Frame 2"
[image15]: ./output_images/test4_output.jpg "Output Frame 3"
[image16]: ./output_images/test5_output.jpg "Output Frame 4"

[video1]: ./project_video_output.mp4 "Project Video"
[video2]: ./challenge_video_output.mp4 "Excerpt from Challenge"
[video3]: ./harder_challenge_video_output.mp4 "Excerpt from Harder Challenge"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

This section goes through the rubric points one at a time and described how they were performed with example outputs, whenever applicable.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  
(You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.)  

This file contains the main write-up. In the folder ´output_images´, a readme.txt also contains some information on the plots found there and the folder structure.
The relevant code files are _P2_AdvancedLaneFinding.py_ (with the single-frame and video pipeline); _P2_subroutines.py_ with the modules/tools which are used to build the pipeline and finally _P2_run.py_ which is an auxiliary script to run a particular image example or apply the pipeline to a video.


### I) Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `calibrateCamera()` function in _P2_subroutines.py_. 

First of all, "object points" which contain the known coordinates of a chessboard in "real world" 3D coordinates are computed. They are the same for all images. 

Then, for each image in the "./camera_cal" folder, the file is read and the corners of the (9, 6) chessboard in image domain (2D coordinates) are detected using `cv2.findChessboardCorners()`. The detected "image points" are gathered to a list.    

Afterwards, all object and image points are input to `cv2.calibrateCamera()`, which returns the distortion coefficients and the camera calibration matrix necessary to undistort the images. These two parameters are saved and applied later by the subroutine `undistort()` in _P2_subroutines.py_, which applies `cv2.undistort()`.

An example calibration image before and after calibration (lens undistortion) is shown below:

|<img width=480px height=0px/> |<img width=480px height=0px/>|
|:-----------------------------:|:---------------------------------------------:|
|![alt text][image1]|![alt text][image2]|

### II) Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The subroutine `undistort()` in _P2_subroutines.py_, applies `cv2.undistort()` using the calibration matrix and the distortion coefficients obtained by the step above. These are stored to a .npz file, the actual calibration is only performed if the file is not found or forced by the user. 

Below an example of the effect of the undistortion in an example image. The effect is more visible at the borders of the image, e.g. the white car.  

![alt text][image3]

This function is called at the pipeline whenever an image is loaded or a frame is input to the processing function. (Cf. _P2_AdvancedLaneFinding.py_: lines 93, 530).

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

A combination of color and gradient thresholds to generate a binary image mask for lane detection is performed in the routine `lanepxmask()`, at lines 290-360 of _P2_subroutines.py_. Tools based on the methods used in the quizzes are used, e.g. `abs_sobel_thresh()`, in the same file. The rationale is to convert from RGB to grayscale/HLS and get the points with high S channel values, high S channel gradients and also high grayscale gradients. Additional steps masking shadows (low L values) are used to increase robustness to shadows and dark patches in the images. (This was not necessary for the test images but added later as testing progressed with the videos). Another aspect which is different from the base algorithm is the use of [morphological operators](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html) to enhance the masks, e.g. by rejecting small clusters of pixels (_morphological opening_) or expanding borders (_dilation_).  

The figure belows illustrates such a mask. The RGB composition has the S channel mask as R, grayscale x-gradient as R and S-channel x-gradient mask as B. The binary output next is shown next. The "ROI restricted" case models the effect of warping to the "bird's eye" perspective and the final plot is the warped mask, as will be explained later on (cf. **II.4**).

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform is implemented in the `Warp2TopDown()` class in _P2_subroutines.py_. The class uses `cv2.getPerspectiveTransform()` alongside the hard-coded source and destination points
````
        self.pts_img = np.float32([[190 + 1, 720], [600 + 1, 445],
                                   [680 - 2, 445], [1120 - 2, 720]])

        self.pts_warp = np.float32([[350, 720], [350, 0], [950, 0], [950, 720]])
````  
to establish the direct and inverse transformation matrices. These are then applies by a call to the `Warp2TopDown.warp()` method. The inverse transform is applied by `Warp2TopDown.unwarp()`. 

The points (listed below) were determined based on "./test_images/straight_lines1.jpg" and "./test_images/straight_lines2.jpg", whose transform is expected to lead to parallel lines. 

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 191, 720      | 350, 720      | 
| 601, 445      | 350, 0        |
| 678, 445      | 950, 0        |
| 1118, 720     | 950, 0        |

Below a verification of the warping using the first image as an example. The red lines correspond to the source and destination points show above, whereas the blue lines are the region of interest (ROI) mentioned in **II.3**. The ROI models the region which will matter for the final result, but is only used for display purposes. 

![alt text][image5]

It should be noted that it was observed that warping **before** the threshold detection led to masks which were less distorted/blurred. Therefore, in the actual pipeline, the order of the steps **II.2** and **II.3** was inverted. Below an example of a threshold binary mask detected _after_ warping. Compare with the mask of **II.2**.

The second plot "RGB image (color mask)" is equal to the first, as no color mask is applied. (This was part of a test which didn't make it into the final version of the pipeline). 

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The tracking of the lane pixels and fitting of a polynomial is done in `find_lane_xy_frommask()` and `find_lane_xy_frompoly()`, both functions in _P2_subroutines.py_. 

The routine `find_lane_xy_frommask()` makes a sliding window search from bottom to top of the mask, starting from a profile of the lower part of the image to locate the pixels which are promissing starting points, as in the quiz. Two noteworthy modifications were made to the logic implemented in the quiz:
- _Region labeling_: First, the most reliable pixels are found using `cv2.morphologyEx()` and `cv2.MORPH_OPEN`, which combined apply the [morphological open](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html) operation, which rejects noise. By applying `cv2.connectedComponents` to this "reliable pixel" mask, connected regions of higher reliability are identified. Their indices are stored and, during the pixel search, it is assured that they are not separated by the sliding window.

- _Window search position update logic_: In the quiz, the center of the next window of each side was determined from the centroid of the pixels in the previous window. This is effective for low curvatures but leads the window to lose track for larger curvatures or wide gaps. Instead, at each step, a fit of the trend of the local windows is performed (unless not enough pixels are gathered so far) and the position of the next window's center is predicted, and this is used as the center of the next iteration. This is done in (_P2_subroutines.py_: lines 640-662) and results in the predictions shown below:   

|<img width=480px height=0px/> |<img width=480px height=0px/>|
|:----------------------------:|:---------------------------:|
|![alt text][image7]           |![alt text][image8]          |

The final output of the sliding window search is shown for two examples below.

|<img width=480px height=0px/> |<img width=480px height=0px/>|
|:----------------------------:|:---------------------------:|
|![alt text][image9]           |![alt text][image10]         |

The routine `find_lane_xy_frompoly()` also fits a polynomial to the detected pixels in the mask, using the _labeling_ technique described above. In this case, however, a-priori coefficients defining the search area are provided for the search. The output of the search, using as input coefficients the coefficients output by `find_lane_xy_frommask()` in the step above are seen in the following plot. 

|<img width=480px height=0px/> |<img width=480px height=0px/>|
|:----------------------------:|:---------------------------:|
|![alt text][image11]          |![alt text][image12]         |


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Once a fit is performed by one of the above methods, coefficients describing x = f(y) for the pixel coordinates are available. These are then converted to coefficients in meters by means of the `cf_px2m()` function in _P2_subroutines.py_, which outputs the coefficients in a new coordinate system in m, for which the origin is at the bottom center of the image and the x and y coordinates are increasing to the right/top, respectively. In this convertion, the pixel to meter ratios given in the project instructions 
In this coordinate system,  
````
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
````
are assumed valid. The radius of curvature is calculated by `r_curve()` in _P2_subroutines.py_ using the formula: 
````
def r_curve(polycoef, y):
    A, B = polycoef[0:2]
    return ((1 + (2 * A * y + B) ** 2) ** 1.5 / np.abs(2 * A))
````
Whether the curve is to the right or left is determined by the sign of the coefficients (positive means to the right in this case). The value and direction of the curvature is annotated to the output frame.

The position of the vehicle with respect to the center is calculated assuming that the camera/image center corresponds to the car's center. In the coordinate system given above, the order zero coefficient of the left and right lane polynomials corresponds to the position of the lane's at the bottom of the image (y_meters = 0). Thus, the average between the two coefficients gives the center of the lane with respect to the car's center. The offset is thus calculated as 
(cf. e.g. _P2_AdvancedLaneFinding.py_: line 376). This is zero for a perfectly center car and _positive_ if the car is to the _left_ of the center of the lane.
````
deltax_center = 0.5*(cf_meters_left[2]+cf_meters_right[2])
````    
The value is also annotated to the output frame.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This step is implemented in the routine `getlane_annotation()` in _P2_subroutines.py_. Given two polynomials, one for each of the lanes, a region in the "bird's eye" view is generated using `cv2.fillPoly()` and warped back into image domain using `Perspective.unwarp()` described in **II.3** 

As described in **II.5**, the curvature and the offset with respect to the lane center are annotated to the output frames. An arrow indicating the direction of the curve, and color-coded to it's intensity is also shown. Below some examples of output frames, computed for the test images.

|<img width=480px height=0px/> |<img width=480px height=0px/>|
|:----------------------------:|:---------------------------:|
|![alt text][image13]          |![alt text][image14]         |
|![alt text][image15]          |![alt text][image16]         |

The annotation adequately represents the lane. The offset w.r.t. the lane center, and the directions and values of the curvature are reasonable. Steeper curves are highlighted by a redish arrow, which is greenish for straight lane sections.

---

### III) Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The final video result builds upon the single frame pipeline by tracking statistics of the frames (notably the mean square error - MSE - of the polynomial fits, computed from the output of `np.polyfit()`, and the number of pixels used for each fit) to judge fit "goodness", which is used in comparing fits across frames. This is implemented into the `LaneLines4Video()` class of _P2_subroutines.py_. A buffer of coefficients for the last few frames (typically 10 were used) is kept, which also allows a comparison of a given frame with its neighbors, smoothing of the output coefficients and a "fall-back" solution in the case of no detection.

The main pipeline is implemented in the `ProcessFrame()` class of _P2_AdvancedLaneFinding.py_. The pipeline tries to get the coefficients from the mask, if the buffer is empty (meaning not enough frames were acquired). If not, it used the medians of the buffer (for each lane) as a first guess for the coefficients and searches the mask around the corresponding region. Whether the output is reliable or not is judged by the MSE and the number of coefficients (cf. `weight_fit_cfs()` in _P2_subroutines.py_: line 500) with respect to the averages of the buffer. If the result is poor, detecting from the mask is attempted. If this new result is judged reliable, it replaces the previous one (independently for each lane side). If both method yields poor results, then the median of the buffer is taken, as a "fall-back solution".

The search for coefficients from the mask in the pipeline also has a "error recovery mode". If the result of the mask search yields polynomials which cross inside the window (not expected, as ideally the lanes are parallel), a coefficient-based search is performed after narrowing down the mask to exclude points which, given the new information, are not part of the lanes. (cf. `ProcessFrame.detectlanes_nopoly()` for details). The polynomial coefficient-based search is the same as in the single-frame pipeline step (cf. **II**).

The annotated video is available [here][video1].

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The new pipeline is based on more advanced computer vision concepts, including the color space transformations, gradient operations and perspective transformations. This added considerably to the robustness, especially in case of changing illumination and poor RGB-contrast conditions. The combination of saturation(S)-channel thresholds and magnitude of the grayscale gradient in the x-direction proved to be a good starting point. The reason is that the S channel is more independent of illumination, which proves to be an important factor for grayscale-based approaches (used in the previous project). The pipeline was improved by adding morphologic operators for mask processing,e.g. to reject noise.    

Frames with "bright asphalt" (light gray, with poor contrast to yellow), dark spots, and especially dark vertical lines (as in the challenge video) are challenging for a pipeline based on these methods, as the dark line has often more contrast to the asphalt than the lane markings itself, and is tracked instead of the lane marking. Some shadows in the asphalt are also seen to have high S values, and thus are not separated from the lanes by the S-threshold alone. In these more demanding frames, e.g. with shadows and dark lines, a masking procedure based on the luminance (L) channel proved to be helpful, though not fail-proof. These allowed at least the first part of the challenge video to be annotated in a somewhat satisfactory manner (the result of the initial pipeline was a complete failure).  

In general, high curvature conditions are also challenging for the lane tracking approach. The window search based on the average of the previous window was found to be "too slow" in predicting the position of the next window center in the x dimension. This was improved by considering the _connected regions_ and the _partial fit_ of the masks' pixels (cf. **II.4**) to track the curvature in a more "aggressive" way, allowing for tracking steeper curves.      

Some of the these difficulties clearly remain. The final version of the pipeline would still tend to lose track of the lanes and track dark lines or asphalt edges instead of the lane markings in case in which the lines are very prominent. The situation of a tunnel under a bridge is tricky due to the proximity of the lane marking, asphalt edge and concrete wall, all of which have similar colors and brightness. Situations with several shadows in the lane are still challenging, and the case of a non-empty lane (i.e. a car or object in front of the car) is not accounted for.
Some of these effects, leading to wobbling lanes, can be viewed in [this exceprt of the challenge video][video2]. The harder challenge case also has in addition the issue of slope (leading to distortion, since the "bird's eye view" maps to a flat road) and bridges whose corners show high contrast to the sky. The first seconds are visible [here][video3]. 


If the project were going to be pursued further, some improvement could probably still be attained by adapting the masks to deal with problematic frames (this would require a deeper analysis of why the frames fail in the challenge videos, given more time). Furthermore, a more robust fitting strategy could be an improvement, considering more of the problematic cases (e.g. curvatures have different signs or change from frame to frame) and trying to use information from other frames or iterative fitting strategies (e.g. with different masks more suitable for a particular case or aiding the fitting by modifying the mask or giving better first guesses for the polynomial search). The implementation of different algorithms, such as the ones pointed out in the "bonus round" of the material would also be a desirable improvement.
