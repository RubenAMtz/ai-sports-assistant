import pandas as pd
import numpy as np
from constants import COLUMNS

# prepare data (reshape frames, concatenate them)
def data_loading(data, cols_to_drop=None):
    """
    Reads all csv's (all frames from a single video in a csv) and cleans missing values (interpolates)
    """
    HEIGHT = 720 # Warning video height!

    single_video_frames = []
    for frame in range(data.shape[0]//25): # 25 data points per frame
        # a chunk are 25 pairs of x,y points. Reshape it to a flat vector of 1,50 and append it to frames
        chunk = data[frame*25:((frame+1)*25), [0,1]]

        chunk[:,1] = HEIGHT - chunk[:, 1]
        chunk = chunk.reshape(1,-1)
        single_video_frames.append(chunk)
  
    svf = np.array(single_video_frames)
    svf = svf.reshape(svf.shape[0], svf.shape[2])
    svf = pd.DataFrame(svf, columns=COLUMNS)
    svf = svf.replace(to_replace=0, value=np.nan)
    svf = svf.drop(columns=cols_to_drop)
    # interpolate col by col when possible (some times values are scarce)
    for col in svf.columns:
        svf[col] = svf[col].interpolate(method='polynomial', order=3).ffill().bfill()
 
    return np.asarray(svf, dtype=np.float32)







def data_distances(data):
    """
    path: dataframe, containing coordinate data
    """
    frames = []
    for frame in data:
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
    # pd.DataFrame(frames.reshape(-1, 276)).to_csv('../test_videos/joint_data_3d_distances.csv')
    return frames.reshape(-1, 276)




def norm(x):
    """MinMaxScale of 'x' (numpy array)
    """
    return (x - x.min()) / (x.max() - x.min())


def normalize_distance_data(data):
    """path: string, to data to transform
    """
    data = norm(data)
    return data.reshape(-1,276)

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)