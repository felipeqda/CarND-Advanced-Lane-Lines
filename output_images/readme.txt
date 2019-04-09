Folder structure:


- For each of the input images, the output with the naming convention <input_image_name>_output.jpg is saved in this folder. The .jpg file is the main output of the pipeline.

- In the folder "calibration", the input and output of the calibration step (correction of lens distortion) is provided for each of the input calibration images. Each output is name "calibration<no>_output.jpg"

- In the folders "<input_image_name>", several .pngs are provided which show the various steps of the pipeline with representative names. The pictures are generated using "single_frame_script()" located in the P2_AdvancedLaneLines.py code file. Refer to the code for details.

- The folder "imageframes&misc" contains additional plots, taken from frames of the videos.