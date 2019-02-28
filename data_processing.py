"""
data_processing.py

File used for data processing the experimental videos for lifting
"""

import pandas as pd
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt
import math
import time
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_X_y
from video import Video, Videos, VideoStateMachine
from constants import COLUMNS, COLUMNS_NORM

# INDEX = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist",
#             "MidHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar",
#             "LEar","LBigToe","LSmallToe","LHeel","RBigToe","RSmallToe","RHeel"]

def pad_video_name(video, format):
    return '{}{}'.format(str(video).zfill(3), format)

# Calculate angles from joints
def calculate_angles(joint_a, joint_b, joint_ref, data):
    """
    joint_a: String, index name e.g. 'LKnee', limit a for angle measurement
    joint_b: String, index name e.g. 'Neck', limit b for the angle measurement
    joint_ref: String, index name e.g. 'LHip', reference point that connects limit a and limit b
    data: DataFrame, contains numeric values for angle computation
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#keypoint-ordering
    """
    # joint_ref - joint_a vector
    x_diff = data.loc[joint_a]['X'] - data.loc[joint_ref]['X']
    y_diff = data.loc[joint_a]['Y'] - data.loc[joint_ref]['Y']
    magnitude = math.sqrt(x_diff**2 + y_diff**2)
    # joint_ref - joint_b vector
    x_diff_ = data.loc[joint_b]['X'] - data.loc[joint_ref]['X']
    y_diff_ = data.loc[joint_b]['Y'] - data.loc[joint_ref]['Y']
    magnitude_ = math.sqrt(x_diff_**2 + y_diff_**2)
    dot_product = (x_diff * x_diff_) + (y_diff * y_diff_)
    return math.degrees(math.acos(dot_product / (magnitude * magnitude_)))

def sad_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def lifting_categories(csv_with_frames):
    """
    csv_with_frames: path, to file containing all normalized frames
    """
    # data filtering?
    data = pd.read_csv(csv_with_frames, index_col=0)
    videos = Videos('number_of_frames_per_video.csv', target='../data_augmentation/joint_data_3d.csv').list
    #print(videos.list[1].neck_y.frame(f=0))
    
    acc_categories = pd.DataFrame(columns=['diff', 'position'])

    for vid in videos:
        category = pd.DataFrame()
        start = 0
        take_off = 0
        power_position = 0
        extension = 0
        reception_snatch = 0
        reception_clean = 0
        jerk = 0
        # iterate through frames of every single video (rows of df)
        sm = VideoStateMachine(vid)
        for frame in range(len(vid.df.values)):
            sm.tick(frame)
            current_state = sm.state
            #print(current_state)
            if current_state == "start":
                start += 1
            elif current_state == "take-off":
                take_off += 1
            elif current_state == "power-position":
                power_position += 1
            elif current_state == "extension":
                extension += 1
            elif current_state == "reception-snatch":
                reception_snatch += 1
            elif current_state == "reception-clean":
                reception_clean += 1
            elif current_state == "jerk":
                jerk += 1
        #break
         # # append every categorized video and create an acc_categories
        print("[{}] Start: {}, Take off {}, Power Position {}, Extension {}, Reception Snatch {}, Reception Clean {}, Jerk {}".format(vid.number, start, take_off, power_position, extension, reception_snatch, reception_clean, jerk))
        
        # try:
        #     acc_categories = acc_categories.append(category, ignore_index=True)    
        # except:
        #     acc_categories = category
        # if count >2: #run for two videos
        #     break
        
    #with pd.option_context('display.max_rows', 300, 'display.max_columns', None):
        # print(df)
        #print(acc_categories.head(160))

    pass

def data_augmentation(data, extra_sets=2):
    """
    data: DataFrame, with shape (# number of frames, 50)
    extra_sets: int, number of datasets to create using the original data and autoencoders
    """
    # rotate pose
    # mirror pose
    # noise to pose
    # translate
    
    # autoencoder
    # clean data
    # scale data
    data = data.values
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    
    if not os.path.isfile('../data_augmentation/joint_data_3d_normalized.csv'):
        pd.DataFrame(data).to_csv('../data_augmentation/joint_data_3d_normalized.csv', columns=COLUMNS_NORM)
    
    X_train, X_test = train_test_split(data, test_size=0.3, random_state=0, shuffle=True)

    if not os.path.isfile('../data_augmentation/X_test.csv'):
        pd.DataFrame(X_test).to_csv('../data_augmentation/X_test.csv')
    
    #X_train, X_test = check_X_y(X_train, X_test)
    # print("BEFORE:", X_train.shape)
    # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    # print("AFTER:", X_train.shape)

    # this is the size of our encoded representations
    for encoding_dim in range(15, 15 + (extra_sets + 1)):
        if not os.path.isfile('../data_augmentation/' + str(encoding_dim) + '_encoded_3d.csv'):

            #encoding_dim = 5 # 20 floats -> compression of factor 4.8, assuming the input is 48 (x and y concatenated)
            input_img = Input(shape=(X_train.shape[1],))
            encoded = Dense(encoding_dim, activation='relu')(input_img)
            decoded = Dense(X_train.shape[1], activation='sigmoid')(encoded)
            autoencoder = Model(input_img, decoded)
            
            encoder = Model(input_img, encoded)
            encoded_input = Input(shape=(encoding_dim,))
            decoder_layer = autoencoder.layers[-1]
            decoder = Model(encoded_input, decoder_layer(encoded_input))

            autoencoder.compile(optimizer='adam', loss='mae')
            autoencoder.fit(X_train, X_train, epochs=150, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

            # as X_test and X_train were shuffled for the training process, 
            # use the original ordered data set "data" to produce ordered outputs
            encoded_vals = encoder.predict(data)
            decoded_vals = decoder.predict(encoded_vals)
            # de-normalize
            #encoded_vals = scaler.inverse_transform(encoded_vals)
            #decoded_vals = scaler.inverse_transform(decoded_vals)
            
            pd.DataFrame(encoded_vals).to_csv('../data_augmentation/' + str(encoding_dim) + '_encoded_3d.csv')
            pd.DataFrame(decoded_vals).to_csv('../data_augmentation/' + str(encoding_dim) + '_decoded_3d.csv')
        
        # print("Encoded: ", encoded_vals)
        # print("Decoded: ", decoded_vals)

        # plt.plot(encoded_vals[0], label="Encoded")
        # plt.show()
        #plt.plot(decoded_vals[0], label="Decoded")
        #plt.plot(X_test[0], label="Original")
        #plt.legend()
        #plt.show()

def compute_metrics():
    # using 2d videos:
        # speed
        # acceleration
        # max height
        # speed from extension to receiving position
        # barbell path

    pass

# COLUMNS = [
#     'nose_x',
#     'nose_y',
#     'neck_x',
#     'neck_y',
#     'rshoulder_x',
#     'rshoulder_y',
#     'relbow_x',
#     'relbow_y',
#     'rwrist_x',
#     'rwrist_y',
#     'lshoulder_x',
#     'lshoulder_y',
#     'lelbow_x',
#     'lelbow_y',
#     'lwrist_x',
#     'lwrist_y',
#     'midhip_x',
#     'midhip_y',
#     'rhip_x',
#     'rhip_y',
#     'rknee_x',
#     'rknee_y',
#     'rankle_x',
#     'rankle_y',
#     'lhip_x',
#     'lhip_y',
#     'lknee_x',
#     'lknee_y',
#     'lankle_x',
#     'lankle_y',
#     'reye_x',
#     'reye_y',
#     'leye_x',
#     'leye_y',
#     'rear_x',
#     'rear_y',
#     'lear_x',
#     'lear_y',
#     'lbigtoe_x',
#     'lbigtoe_y',
#     'lsmalltoe_x',
#     'lsmalltoe_y',
#     'lheel_x',
#     'lheel_y',
#     'rbigtoe_x',
#     'rbigtoe_y',
#     'rsmalltoe_x',
#     'rsmalltoe_y',
#     'rheel_x',
#     'rheel_y'
#     ]

# COLUMNS_NORM = [
#     'nose_x',
#     'nose_y',
#     'neck_x',
#     'neck_y',
#     'rshoulder_x',
#     'rshoulder_y',
#     'relbow_x',
#     'relbow_y',
#     'rwrist_x',
#     'rwrist_y',
#     'lshoulder_x',
#     'lshoulder_y',
#     'lelbow_x',
#     'lelbow_y',
#     'lwrist_x',
#     'lwrist_y',
#     'midhip_x',
#     'midhip_y',
#     'rhip_x',
#     'rhip_y',
#     'rknee_x',
#     'rknee_y',
#     'rankle_x',
#     'rankle_y',
#     'lhip_x',
#     'lhip_y',
#     'lknee_x',
#     'lknee_y',
#     'lankle_x',
#     'lankle_y',
#     'reye_x',
#     'reye_y',
#     'leye_x',
#     'leye_y',
#     'rear_x',
#     'rear_y',
#     'lbigtoe_x',
#     'lbigtoe_y',
#     'lsmalltoe_x',
#     'lsmalltoe_y',
#     'lheel_x',
#     'lheel_y',
#     'rbigtoe_x',
#     'rbigtoe_y',
#     'rsmalltoe_x',
#     'rsmalltoe_y',
#     'rheel_x',
#     'rheel_y'
#     ]

def data_loading(path, clean=True, cols_to_drop=None):
    """
    Reads all csv's (all frames from a single video in a csv) and cleans missing values (interpolates)
    """
    if not os.path.isfile('../data_augmentation/joint_data_3d.csv'):
        os.chdir(path)
        files = os.listdir('./')
        #print(files)
        #frames =[]
        frames = pd.DataFrame()

        single_video_frames = []
        num_of_frames = []
        for file in tqdm.tqdm(files):
            data = pd.read_csv(file)
            num_of_frames.append(data.shape[0]//25)
            for frame in range(data.shape[0]//25): # 25 data points per frame
                # a chunk are 25 pairs of x,y points. Reshape it to a flat vector of 1,50 and append it to frames
                chunk = data.iloc[frame*25:((frame+1)*25),[1,4]]
                chunk = chunk.values.reshape(1,-1)
                single_video_frames.append(chunk)
            if clean:
                svf = np.array(single_video_frames)
                svf = svf.reshape(svf.shape[0], svf.shape[2])
                
                svf = pd.DataFrame(svf, columns=COLUMNS)
                svf = svf.replace(to_replace=0, value=np.nan)
                svf = svf.drop(columns=cols_to_drop)
                # interpolate col by col when possible (some times values are scarce)
                for col in svf.columns:
                    # try:
                    svf[col] = svf[col].interpolate(method='polynomial', order=3).ffill().bfill()
                    # except ValueError:
                    #     pass
                single_video_frames = []
            try:
                frames = frames.append(svf, ignore_index=True)
            except:
                frames = svf
        # frames = frames.drop(columns=cols_to_drop)
        frames.to_csv('../data_augmentation/joint_data_3d.csv')
        return np.asarray(frames, dtype=np.float32)

#print(len(data)//2)
#print(data.shape)
#print(data.iloc[[0]].values.reshape(25,2))

data_loading('../json_to_csv_3d', cols_to_drop = ['lear_x','lear_y'])

data = pd.read_csv('../data_augmentation/joint_data_3d.csv', index_col=0)
data_augmentation(data, extra_sets=5)

frames_per_video = pd.read_csv('./number_of_frames_per_video.csv', index_col=0)
#print(frames_per_video)

lifting_categories('../data_augmentation/joint_data_3d_normalized.csv')

#print(data.values.reshape(len(data)//2,25))

#data_df = pd.DataFrame(data, columns=INDEX)
#data.to_csv('csvs_ready_to_train.csv')

# hip_angle = calculate_angles('Neck','LKnee', 'LHip', data)
# knee_angle = calculate_angles('LHip','LAnkle','LKnee', data)  