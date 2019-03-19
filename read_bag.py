#####################################################
##               Read bag from file                ##
#####################################################
import pyrealsense2 as rs
import numpy as np
import cv2
import os.path
import tqdm

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
width = 1280
height = 720
fps = 30

try:
    # Streaming loop
    PATH = 'D:/Experiment/bag files'
    # for filename in tqdm.tqdm(os.listdir(PATH)):
        # out = cv2.VideoWriter('../videos-3d/Depth/'+ filename[:3] + '.avi',fourcc, fps, (width,height))
    filename = '109.bag'
    out = cv2.VideoWriter('../processed-videos-3d/'+ filename[:3] + '.avi', fourcc, fps, (width,height))
    # Create pipeline
    pipeline = rs.pipeline()
    # Create a config object
    config = rs.config()
    rs.config.enable_device_from_file(config, PATH + "/" + filename)
    # Configure the pipeline to stream the depth stream
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    # Configure the pipeline to stream the infrared stream
    config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)
    #config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)
    # Start streaming from file
    pipeline.start(config)
    count = 0
    while True:
        # Get frameset of depth
        frames = pipeline.wait_for_frames()
        # initialize frame_number
        if count == 0:
            frame_number = frames.get_frame_number()
        # compare the value of current frame number vs first frame number saved, if true update value, else break
        if frames.get_frame_number() >= frame_number:
            frame_number = frames.get_frame_number()
        else:
            pipeline.stop()
            break

        # Get depth frame
        depth_frame = frames.get_depth_frame()
        ir_frame = frames.get_infrared_frame(2)
        
        # Colorize depth frame to jet colormap
        depth_color_frame = rs.colorizer().colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        # Convert ir_frame to numpy array to render image in opencv
        ir_image = np.asanyarray(ir_frame.get_data())

        ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
        frame = cv2.flip(ir_image, 0)            
        # out.write(depth_color_image)
        out.write(ir_image)
        
        # Render image in opencv window
        #cv2.imshow("Depth Stream", ir_image)
    
        #cv2.imwrite("../processed-videos/" + filename[0:3] + "/_depth_" + str(count) + ".png", depth_color_image)
        #cv2.imwrite("../processed-videos/" + filename[0:3] + "/_infrared_" + str(count) + ".png", ir_image)
        
        #cv2.imshow('frame', ir_image)
        count += 1

    out.release()

finally:
    pass