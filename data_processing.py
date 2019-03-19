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
# from keras.layers import Input, Dense
# from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_X_y
from video import Video, Videos, VideoStateMachine
from constants import COLUMNS, COLUMNS_NORM, BODY_PAIRS
from xml2dataframe import XML2DataFrame
import cv2
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





def lifting_categories():
    """
    csv_with_frames: path, to file containing all normalized frames
    """
    
    video_list = pd.read_csv('../classification/grading_scheme 3d.csv')
    video_list = video_list['Unnamed: 0']
    video_list.name = 'videos'
    #print(video_list[0])

    videos = Videos('num_frames_per_video.csv', target='../data_augmentation/joint_data_3d.csv').list
    
    acc_categories = pd.DataFrame(columns=['diff', 'position'])
    count = 0
    category = []
    frames_ = []
    video_ = []
    for i, vid in enumerate(videos):
        # category = pd.DataFrame()
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
            category.append(current_state)
            frames_.append(frame)
            video_.append(video_list[i])

        #break
         # # append every categorized video and create an acc_categories
        print("[{}] Start: {}, Take off {}, Power Position {}, Extension {}, Reception Snatch {}, Reception Clean {}, Jerk {}".format(vid.number, start, take_off, power_position, extension, reception_snatch, reception_clean, jerk))
    video_name = pd.DataFrame(video_, index=frames_, columns=['video_name'])
    category = pd.DataFrame(category, index=frames_, columns=['category'])
    pd.concat([video_name,category], axis=1).to_csv('../classification/lifting_categories.csv')
    
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
            # encoded_vals = scaler.inverse_transform(encoded_vals)
            decoded_vals = scaler.inverse_transform(decoded_vals)
            
            # pd.DataFrame(encoded_vals).to_csv('../data_augmentation/' + str(encoding_dim) + '_encoded_3d.csv')
            pd.DataFrame(decoded_vals).to_csv('../data_augmentation/' + str(encoding_dim) + '_decoded_3d.csv')
        
    
# from lxml import objectify




def compute_metrics():
    # using 2d videos:
        # dir: ../trajectories/2d/*.xml
        # speed
        # acceleration
        # max height
        # speed from extension to receiving position
        # barbell path
    path = '../trajectories/2d/'
    filenames = os.listdir(path)
    # df_2d holds a list of DataFrame objects: [df1, df2, ..., dfn]
    df_2d = []
    # for filename in filenames:
        # df_2d.append(XML2DataFrame(path, filename).to_df())
    df_2d.append(XML2DataFrame(path, filenames[0]).to_df())
    #print(df_2d[0])
    print(len(df_2d[0]))
    print(df_2d[0].columns)
    velocity = []
    barbell = pd.DataFrame()
    for df in df_2d:
        for frame in range(len(df)):
            if frame > 0:
                pass
                # print("frame: ", frame)
                #print(deltadistance(df['y'][frame-1],df['y'][frame])/deltatime(df['t'][frame-1], df['t'][frame]))
            #else:
            #    barbell.loc[frame, 'y']

def deltatime(time1, time2):
    """ Calculates time difference between 2 datetime.time objects
        in seconds

        Returns a float number (not a datetime.time object)
    """
    t1 = time1.minute * 60 + time1.second + time1.microsecond  / 1000000
    t2 = time2.minute * 60 + time2.second + time2.microsecond / 1000000
    delta = t2 - t1
    return abs(delta)



def deltadistance(distance1, distance2):
    """ Calculates distance difference between 2 floats objects

        Returns a float number (always positive)
    """
    return abs(distance2 - distance1)


def speed(delta_distance, delta_time):
    return delta_distance / delta_time

def delta_speed(speed1, speed2, delta_time):
    return abs(speed1 - speed2) / delta_time

def acceleration(delta_speed, delta_time):
    return delta_speed / delta_time






def data_loading(path, clean=True, cols_to_drop=None):
    """
    Reads all csv's (all frames from a single video in a csv) and cleans missing values (interpolates)
    """
    HEIGHT = 720
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
        frames.to_csv('../../data_augmentation/joint_data_3d.csv')
        return np.asarray(frames, dtype=np.float32)




def frames_per_video(path):
    files = os.listdir(path)
    print(files)
    num_of_frames = []
    for file in tqdm.tqdm(files):
        data = pd.read_csv(path + '/' +  file)
        num_of_frames.append(data.shape[0]//25)
    frames = pd.DataFrame(num_of_frames, columns=['frames'])
    frames.to_csv('./num_frames_per_video.csv')

def rotation_on_self(angle):
    return cv2.getRotationMatrix2D((1280/2, 720/2), angle, 1.0) # pylint: disable=maybe-no-member





def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)




# new_df = pd.DataFrame()
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
    pd.DataFrame(frames.reshape(-1, 276)).to_csv('../data_augmentation/joint_data_3d_distances_' + path[-7:-4] + '.csv')
    return pd.DataFrame(frames.reshape(-1, 276))





def grades_and_categories(categories_path, grades_path):
    if not os.path.isfile('../classification/grades_and_categories.csv'):    
        categories = pd.read(categories_path)
        grades = pd.read_csv(grades_path)
        categories.rename(index=str, columns={'Unnamed: 0': 'frame_#'}, inplace=True)
        grades.rename(index=str, columns={'Unnamed: 0': 'video_name'}, inplace=True)

        grades_= []
        # for i, video in enumerate(grades['video_name']):
        for j, video_ in enumerate(tqdm.tqdm(categories['video_name'])):
            #filter by video name
            video_name = video_
            category = categories['category'][j]
            # print(video_name, category)
            filtered_grade = grades[grades.video_name == video_name]
            if category == 'reception-snatch' or category == 'reception-clean':
                filtered_grade = filtered_grade[category[:9]][0]
            else:
                filtered_grade = filtered_grade[category][0]
            grades_.append(filtered_grade)
            
        categories['grade'] = grades_
        categories.to_csv('../classification/grades_and_categories.csv')


def rotate_around_y(angle):
    """ Rotate points (x,y) in joint_data_3d.csv by a given angle
        Returns a DataFrame object with the values rotated
    """
    data = pd.read_csv('../data_augmentation/joint_data_3d.csv')
    data.rename(index=str, columns={'Unnamed: 0': 'frame_num'}, inplace=True)
    HEIGHT = 720
    WIDTH = 1280
    tx = WIDTH / 2
    ty = HEIGHT / 2
    new_frames = np.empty(0)
    # transformation matrix
    mt1 = np.array([1, 0, -tx, 0, 1, -ty, 0, 0, 1]).reshape(3,3)
    mr = np.array([np.cos(np.radians(angle)), 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3,3)
    mt2 = np.array([1, 0, tx, 0, 1, ty, 0, 0, 1]).reshape(3,3) 
    # iterate through rows
    for frame in tqdm.tqdm(data.frame_num):
        # extract elements from row and reshape them
        coordinates = data.iloc[frame][1:].values.reshape(24,-1)
        new_coordinates = np.empty(0)
        # iterate through each element
        for element in coordinates:
            # prepare the data to be multiplied (append a 1)
            element = np.append(element, 1)
            # translate to the origin
            trans = np.matmul(mt1, element)
            # rotate
            rot = np.matmul(mr, trans)
            # translate back to original position
            new_coordinates = np.append(new_coordinates, np.matmul(mt2, rot)[:-1])
        new_frames = np.append(new_frames, new_coordinates)
    return pd.DataFrame(new_frames.reshape(-1,48), columns=COLUMNS_NORM)

# ROTATIONS = [45, 135, 180]
# for angle in ROTATIONS:
#     transformed_data = rotate_around_y(angle)
#     transformed_data.to_csv('../data_augmentation/transformed_data_'+ str(angle) + '.csv')

def compare_original_transformed(frame):
    original_data = pd.read_csv('../data_augmentation/joint_data_3d.csv', index_col=0)
    transformed_data = pd.read_csv('../data_augmentation/transformed_data_45.csv', index_col=0)
    # select first frame from original and transformed data
    frame_o = original_data.iloc[frame]
    frame_t = transformed_data.iloc[frame]
    # get x and y values from original frame and transformed frame
    x_o, y_o = get_xy_from_frame(frame_o)
    x_t, y_t = get_xy_from_frame(frame_t)
    
    show_lines(x_o, y_o, 'ro-')
    show_lines(x_t, y_t, 'bo-')
    plt.show()




def show_lines(x, y, color):
    for i, e in enumerate(BODY_PAIRS):
        p0 = e[0]
        p1 = e[1]
        if e[0] >= 18:
            p0 = p0 - 1
        if e[1] >= 18:
            p1 = p1 - 1
        
        x1, y1 = x[p0], y[p0]
        x2, y2 = x[p1], y[p1]
        draw_line(x1, y1, x2, y2, color)
    # plt.show()

def draw_line(x1, y1, x2, y2, color):
    x = [x1, x2]
    y = [y1, y2]
    plt.plot(x, y, color)
    # plt.show()


def get_xy_from_frame(frame_data):
    frame = frame_data.values.reshape(24, -1)
    x = np.empty(0)
    y = np.empty(0)
    for element in frame:
        x = np.append(x, element[0])
        y = np.append(y, element[1])
    return x, y





def norm(x):
    """MinMaxScale of 'x' (numpy array)
    """
    return (x - x.min()) / (x.max() - x.min())



def normalize_distance_data(path):
    data = pd.read_csv(path, index_col=0)
    cols = data.columns
    video_and_frames = pd.read_csv('../classification/grades_and_categories.csv')
    num_of_frames = video_and_frames['frame_#'].values
    video_name = video_and_frames['video_name'].values
    data = data.values
    videos, count = np.unique(video_name, return_counts=True)
    # print(norm(data[:count[0]]))
    # print(norm(data[:count[0]]).min(), norm(data[:count[0]]).max())

    acc = 0
    vids = np.empty(0)
    for i in tqdm.tqdm(count):
        vid_distances = data[acc:i+acc]
        vids = np.append(vids, norm(vid_distances))
        acc += i
    
    pd.DataFrame(vids.reshape(-1,276), columns=cols).to_csv('../data_augmentation/joint_data_3d_distances_180_norm.csv')

normalize_distance_data('../data_augmentation/joint_data_3d_distances_180.csv')
# data_distances('../data_augmentation/transformed_data_180.csv')

# compare_original_transformed(54)
# 
# a = plt.figure()
# draw_line(1, 2, 3, 4)
# plt.show()



# print(np.matmul(rotate_around_y(45), np.array([1, 0, 1])))
# grades_and_categories('../classification/lifting_categories.csv', '../classification/grading_scheme 3d.csv')
# frames_per_video('../openpose-api/bodykeypoints')

#print(len(data)//2)
#print(data.shape)
#print(data.iloc[[0]].values.reshape(25,2))

# data_loading('../openpose-api/bodykeypoints', cols_to_drop = ['lear_x','lear_y'])

# data = pd.read_csv('../data_augmentation/joint_data_3d.csv', index_col=0)
# data_augmentation(data, extra_sets=5)

# frames_per_video = pd.read_csv('./number_of_frames_per_video.csv', index_col=0)



#print(frames_per_video)

# lifting_categories()
    
# data_distances(new_df)

# def 


# compute_metrics()

#print(data.values.reshape(len(data)//2,25))

#data_df = pd.DataFrame(data, columns=INDEX)
#data.to_csv('csvs_ready_to_train.csv')

# hip_angle = calculate_angles('Neck','LKnee', 'LHip', data)
# knee_angle = calculate_angles('LHip','LAnkle','LKnee', data)  

"""
Results to present:
- Ground truth of lifting categories vs algorithm (just learn from the data)
- Ground truth of barbell metrics vs algorithm (we need to do tracking)
- Ground truth of assessment vs algorithm (just learn from data)
- Ground truth of barbell path vs perspective correction results ??
"""