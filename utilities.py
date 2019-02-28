"""
utilities.py

File with functions commonly used in the project
"""
import os
import subprocess
import pandas as pd
import numpy as np
import math
import tqdm

FILE_NUMBER = 108

def main():
    for i in range(1,FILE_NUMBER+1):
        path = '../processed-videos-3d/videos/'+ str(i).zfill(3)
        if not os.path.exists(path):
            os.mkdir(path)

def cut_video(input_filename, time_window, folder, output_filename, output_path=None):
    if output_path is None:
        output_path = "D:/Experiment/processed-videos-2d/"
    if os.path.isfile("{}/{}/{}".format(output_path, folder, output_filename)):
        return
    # folder where videos to be cut are
    os.chdir('../videos-2d/')
    args = [
        'ffmpeg',
        '-loglevel', 'panic',
        '-i', input_filename,
        '-ss', time_window[0],
        '-to', time_window[1],
        # '-c', 'copy', "cut_{}".format(filename)
        '-async', '1',
        "{}/{}/{}".format(output_path, folder, output_filename)
    ]
    #print(args)
    subprocess.call(args)

def pad_video_name(video, format):
    return '{}{}'.format(str(video).zfill(3), format)

def ffmpeg_time(time):
    if len(time) == 1:
        time = time + ".0"
    try:
        if time[2] != '.':
            time = '0' + time
    except:
        time = time + ".0"
    time = time.ljust(6, '0')
    if len(time) >= 7:
        time = '00:' + time
    else:
        time = "00:00:" + time
    return time

def create_json(video_read_path, video_save_path, json_path):   
    DIR = "D:/Experiment/processed-videos-2d/"
    # folder where openposedemo.exe is to be executed at
    os.chdir('C:/Users/ruben/Documents/Github/openpose')

    # save json files relative to DIR
    if not os.path.isdir(json_path):
        os.makedirs(DIR + json_path)
    
    args = [
        'OpenPoseDemo',
        "--video", DIR + video_read_path,
        "--write_video", DIR + video_save_path,
        "--write_json", DIR + json_path,
        #"--net_resolution", "1312x736"
        #"--scale_number", "4",
        #"--scale_gap", "0.25",
        #"--display", "0",
        "--number_people_max", "1"
        ]
    subprocess.call(args)

def json_to_csv(path):
    """
    path is an updatable reference to search for the json files, in this case "json_save"
    """
    # Image properties:
    WIDTH = 1280
    HEIGHT = 720
    DIR = '../processed-videos-3d/videos/' # path parameters is passed relative to this DIR
    files = os.listdir(DIR + path)
    dataf = pd.DataFrame(columns=['X','Y','Confidence_Factor','Y_rect']) 
    index = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist",
            "MidHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar",
            "LEar","LBigToe","LSmallToe","LHeel","RBigToe","RSmallToe","RHeel"]
    for file in files:
        data = pd.read_json(DIR + path +"/" +  file)
        data = np.array(data['people'][0]['pose_keypoints_2d'])
        data = data.reshape(-1,3)
        data = pd.DataFrame(data, columns=['X','Y','Confidence_Factor'], index=index)
        # To correct for picture coordinates as 0,0 is top left corner
        data[['Y_rect']] = HEIGHT - data[['Y']]
        # Some values are actually zero, leave them as such
        for i, e in enumerate(data['Y_rect']):
            if e == HEIGHT:
                data.loc[index[i],'Y_rect'] = 0
        dataf = pd.concat([dataf, data])
    dataf.to_csv('../json_to_csv_3d/'+ path[-5:] +'.csv')

#iterate through video list
def csv_to_args(data, json=False, csv=False, video_cut=False):
    """
    Utility method that feeds to functions such as create_json, json_to_csv or cut_video
    -
    data: DataFrame, containing the cut times per video e.g. 
                    data = pd.read_csv('./video-split-data-3d.csv', dtype=str, na_filter=False)
    json: method, calls openpose and generates json based on the joints
    csv: method, reads json files and converts them into csv
    video_cut: method, cut videos into parts given cut times
    """

    files_created = []
    for v in tqdm.tqdm(range(len(data['video']))):
        video = data['video'][v]
        #video format
        video = pad_video_name(video, '.mp4')
        time_cols = data.columns[data.columns != 'video']
        # iterate through time columns
        instance = 0
        
        for start in [1, 3, 5, 7]:
            if data.iloc[v][start] != "":
                instance += 1
                start_t = data.iloc[v][start]
                stop_t = data.iloc[v][start + 1]
                folder = video[:3]
                video_result = video[:3] + '_' + str(instance) + '.mp4'
                files_created.append(video_result)
                
                video_save =  folder + "/" + video_result[0:5] + ".avi"
                json_save = folder + "/" + 'json/' + video_result[0:5]
                if json:
                    #print("video_read_path:", folder + "/" + video_result, "- video_save_path:", folder + "/" + video_result[0:5] + ".avi", 
                    #"- write_jason:", folder + "/" + 'json/'+ video_result[0:5])
                    video_read = folder + "/" + video_result
                    create_json(video_read, video_save, json_save)
                if csv:
                    json_to_csv(json_save)
                if video_cut:
                    #print("input:",video, "- folder:", folder, "- output:", video_result, ffmpeg_time(start_t), ffmpeg_time(stop_t))
                    cut_video(video, (ffmpeg_time(start_t), ffmpeg_time(stop_t)), folder, video_result)
                
scheme = pd.DataFrame(columns=['posicion_inicial','despegue','power_position','extension','recepcion','jerk (clean)'], index=files_created)
#scheme.to_csv('./grading_scheme.csv',index=True)

# #file with timestamp data :
data = pd.read_csv('./video-split-data.csv', dtype=str, na_filter=False) #do not parse na values
#csv_to_args(data=data, json=True)