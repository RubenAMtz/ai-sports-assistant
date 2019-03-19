import os
import subprocess
import pandas as pd
import numpy as np
import math
import tqdm
from constants import COLUMNS, COLUMNS_NORM
from joblib import load, dump
import sys
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



# Import Openpose (Windows/Ubuntu/OSX)
PYOPENPOSE_DIR = "C:/Users/ruben/Documents/Github/openpose/build/python/openpose/Release"
MODELS_DIR = "C:/Users/ruben/Documents/Github/openpose/models/"
KEYPOINTS = ['nose', 'neck', 'rshoulder', 'relbow', 'rwrist', 'lshoulder','lelbow', 
            'lwrist', 'midhip', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 'lankle',
            'reye', 'leye', 'rear', 'lear', 'lbigtoe', 'lsmalltoe', 'lheel', 'rbigtoe',
            'rsmalltoe', 'rheel']

sys.path.append(PYOPENPOSE_DIR)
import pyopenpose as op




def cut_video(input_filename, time_window, output_filename, output_path=None):
    """
    time window: list, format of time is 00:00:00.xxx
    """
    args = [
        'ffmpeg',
        '-loglevel', 'panic',
        '-i', input_filename,
        '-ss', time_window[0],
        '-to', time_window[1],
        # '-c', 'copy', "cut_{}".format(filename)
        '-async', '1',
        "{}/{}".format(output_path, output_filename)
    ]
    #print(args)
    subprocess.call(args)





#############################
## create csv out of openpose
#############################
params = {
    'model_folder': MODELS_DIR,
    'number_people_max': 1
}



# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()





def run_openpose(frame):

    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    return datum.poseKeypoints.copy()





def joints_to_csv(video_path, csv_path):
        
    cap = cv2.VideoCapture(video_path)
    bodykeypoints = []

    while (True):    
        ret, image = cap.read()
        if not ret:
            break
        bodykp = run_openpose(image)
        bodykeypoints.append(bodykp.copy())
        
    bodykps = pd.DataFrame(columns=['x','y','confidence'])
    for frame in bodykeypoints:
        # print(frame[0])
        bodykp = pd.DataFrame(frame[0], columns=['x','y','confidence'], index= KEYPOINTS)
        bodykps = bodykps.append(bodykp)
        bodykps.to_csv(csv_path)






# prepare data (reshape frames, concatenate them)
def data_loading(path, clean=True, cols_to_drop=None):
    """
    Reads all csv's (all frames from a single video in a csv) and cleans missing values (interpolates)
    """
    HEIGHT = 720
    if not os.path.isfile('../test_videos/joint_data_3d.csv'):
        # os.chdir(path)
        files = os.listdir(path)
        print(files)
        #frames =[]
        frames = pd.DataFrame()

        single_video_frames = []
        num_of_frames = []
        # 001_01.csv, ... 
        for file in tqdm.tqdm(files):
            data = pd.read_csv(path + file)
            num_of_frames.append(data.shape[0]//25)
            for frame in range(data.shape[0]//25): # 25 data points per frame
                # a chunk are 25 pairs of x,y points. Reshape it to a flat vector of 1,50 and append it to frames
                chunk = data.iloc[frame*25:((frame+1)*25),[1,2]]
                chunk[['y']] = HEIGHT - chunk[['y']]
                # print(chunk)
                # break
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
        frames.to_csv('../test_videos/joint_data_3d.csv')
        return np.asarray(frames, dtype=np.float32)





def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)





def data_distances(path):
    """
    path: dataframe, containing coordinate data
    """
    data = pd.read_csv(path, index_col=0)
    data = data.values
    frames = []
    for frame in tqdm.tqdm(data):
        distances = np.empty(0)
        for i in range(len(frame)):
            #first pair (reference)
            if ((i*2)+1) == (len(frame)-1):
                break
            for j in range(len(frame)):
                # moving pair to be multiplied with reference
                if (i*2+3+j*2) <= (len(frame)-1):
                    distances = np.append(distances, calculate_distance(frame[i*2], frame[i*2+1], frame[(i*2)+2+j*2],frame[(i*2)+3+j*2]))
                else:
                    break
        frames.append(distances)
    frames = np.array(frames)
    pd.DataFrame(frames.reshape(-1, 276)).to_csv('../test_videos/joint_data_3d_distances.csv')
    return pd.DataFrame(frames.reshape(-1, 276))




def norm(x):
    """MinMaxScale of 'x' (numpy array)
    """
    return (x - x.min()) / (x.max() - x.min())




def normalize_distance_data(path):
    """path: string, to data to transform
    """
    data = pd.read_csv(path, index_col=0)
    cols = data.columns
    data = norm(data.values)
    pd.DataFrame(data.reshape(-1,276), columns=cols).to_csv('../test_videos/joint_data_3d_distances_norm.csv')



video_name = 'lydia.mp4'
file_name = 'D:/Experiment/test_videos/' + video_name
output_path = 'D:/Experiment/test_videos/processed_videos'
time_window = ['00:00:16.25', '00:00:24']

print("Cutting video...")
# cut_video(file_name, time_window, video_name, output_path) #done

bodykeypoints_path = 'D:/Experiment/test_videos/bodykeypoints/'
videos_path = 'D:/Experiment/test_videos/processed_videos/'

print("Generation joints...")
# joints_to_csv(videos_path + video_name, bodykeypoints_path + 'data_{}.csv'.format(video_name[:-4]))

print("Reshaping joints...")
# data_loading(bodykeypoints_path, cols_to_drop = ['rear_x','rear_y'])
data_loading(bodykeypoints_path, cols_to_drop = ['lear_x','lear_y'])

print("Calculating distances...")
data_distances('../test_videos/joint_data_3d.csv')

print("Normalizing distances...")
normalize_distance_data('../test_videos/joint_data_3d_distances.csv')

print("Inference...")
# import X data
#X = pd.read_csv('../data_augmentation/joint_data_3d.csv', index_col = 0)
query = pd.read_csv('../test_videos/joint_data_3d_distances_norm.csv', index_col = 0)

# scaler = load('models/models with distances/scaler_not_augmented.joblib')
scaler = load('models/scaler.joblib')

query = scaler.transform(query)

classifier = load('./models/XGBClassifier.joblib')

pred = classifier.predict(query)

print(pred)


