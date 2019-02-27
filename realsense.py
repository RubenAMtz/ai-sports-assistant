## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
width = 1280
height = 720
fps = 30
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
# records the video in bag format (native format of the SDK)
config.enable_record_to_file('test.bag')
# Use rs-converter.exe to convert to csv:
# rs-convert.exe -v test -i test.bag

# Configure a video writer object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi',fourcc, fps, (width,height))

# Start streaming
pipeline.start(config)

# To count number of frames grabbed
counter = 0

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Save color image to video
        out.write(color_image)
        
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))
        #images = np.hstack((color_image, depth_image))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)
        #if key in (27, ord("q")):
        #    break

        if counter>=150: # change it to record what length of video you are interested in 
            print("Done!") 
            break
        
        counter += 1

finally:

    # Stop streaming
    pipeline.stop()