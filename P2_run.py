
INTERACTIVE_MODE = True #set to False for pure video processing, True for testing/debugging

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

# modules
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from pdb import set_trace as stop  # useful for debugging
import glob  # to expand paths
from moviepy.editor import VideoFileClip
from P2_AdvancedLaneFinding import single_frame_analysis,ProcessFrame


"""------------------------------------------------------------"""
""" Calibration images                                         """
"""------------------------------------------------------------"""
# run to get calibration images
# _,_ = calibrateCamera(FORCE_REDO=True)


"""------------------------------------------------------------"""
""" Single-frame test image examples                           """
"""------------------------------------------------------------"""
# image_list = os.listdir('./test_images')
image_list = ['straight_lines1.jpg',
              'straight_lines2.jpg',
              'test1.jpg',
              'test2.jpg',  # left curve
              'test3.jpg',
              'test4.jpg',  # shadow in lane
              'test5.jpg',
              'test6.jpg']

#### Run for single input
#input_image = "test_images" + os.sep + image_list[5] #5!  #indices [0-7]
#single_frame_analysis(input_image,
#                      SHOW_CALIBRATION=False, SHOW_COLOR_GRADIENT = False, SHOW_WARP = False,  SHOW_FIT = True)
#raise Exception('stop')

#### Run for all
#for input_image in image_list: single_frame_analysis("test_images" + os.sep + input_image)
#stop()

"""------------------------------------------------------------"""
""" Video processing                                           """
"""------------------------------------------------------------"""
test_videos = ['project_video.mp4', 'challenge_video.mp4', 'harder_challenge_video.mp4']
input_video = test_videos[0] # choose video
output_video = os.path.basename(input_video).split('.mp4')[0]+'_output.mp4'

# processing flags
QUICK_TEST = False
t_beg, t_end = (0, 4)  #(20, 24) for projectvideo
PROCESS_VIDEO = False


# applies process_image frame by frame
processing_pipeline = ProcessFrame(N_buffer = 10)
print('')
print('Input video: '+input_video)
# choose input file and range
if QUICK_TEST:
    #take a short sub-clip for the testing
    clip_in = VideoFileClip(input_video).subclip(t_beg, t_end)
    print('Processing only an excerpt...')
else:
    clip_in = VideoFileClip(input_video)
# process video
if PROCESS_VIDEO:
    clip_out = clip_in.fl_image(processing_pipeline)
    clip_out.write_videofile(output_video, audio=False)

# BENCHMARKS
# 1) project video
#single_frame_analysis(clip_in.get_frame(41.801), SHOW_CALIBRATION=False, SHOW_COLOR_GRADIENT = True, SHOW_WARP = False,  SHOW_FIT = True)
# plt.imshow(processing_pipeline(clip_in.get_frame(41.801)))
# take video inputs from (20, 24) s!
single_frame_analysis(clip_in.get_frame(40.6), SHOW_CALIBRATION=False, SHOW_COLOR_GRADIENT = True, SHOW_WARP = False,  SHOW_FIT = True)
# plt.imshow(processing_pipeline(clip_in.get_frame(41.801)))


# 2) challenge video
#single_frame_analysis(clip_in.get_frame(0.0), SHOW_COLOR_GRADIENT = True, SHOW_WARP = False,  SHOW_FIT = True)
# plt.imshow(processing_pipeline(clip_in.get_frame(0)))



# USEFUL COMMANDS:
# %matplotlib qt5
# NOTE: get frame by <no>
# clip_in.get_frame(<no>*1.0/clip_in.fps)
