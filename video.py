# pylint: disable=no-member
"""
video.py

File used to help categorize videos
"""

import pandas as pd
import numpy as np


class Video(object):

    def __init__(self, df, number=None, avoid=False):
        self.number = number
        self.df = pd.DataFrame(df)
        if avoid:
            return
        
        for col in self.df.columns:
            setattr(self, col, Video(self.df[col], avoid=True))

    def frame(self, f):
        return self.df.iloc[f]
    
    def __str__(self):
        return str(self.df)

class VideoStateMachine(object):
    
    def __init__(self, video):
        self.state = "start"
        self.state_method = self.state0
        self.video = video
        
    def tick(self, frame):
        if self.state_method:
            self.state_method = self.state_method(frame)

    def state0(self, frame):
        self.state = "start"
        
        midhip_y_all = self.video.midhip_y.df.values
        
        # condition to jump to start
        if ((midhip_y_all[frame] - midhip_y_all[0]) <= midhip_y_all[0] * 0.01):
            return self.state1
        else:
            return self.state0
        
    def state1(self, frame):
        self.state = "start"

        rwrist_y_all = self.video.rwrist_y.df.values
        
        # condition to jump to take off
        if ((rwrist_y_all[frame] - rwrist_y_all[0]) > rwrist_y_all[0] * 0.01):
            return self.state2
        else:
            return self.state1
    
    def state2(self, frame):
        self.state = "take-off"
        
        midhip_y_all = self.video.midhip_y.df.values
        rwrist_y_all = self.video.rwrist_y.df.values
        rknee_y_all = self.video.rknee_y.df.values

        # condition to jump to power position
        if (rwrist_y_all[frame] > rknee_y_all[frame] and rwrist_y_all[frame] < midhip_y_all[frame]):
            return self.state3
        else:
            return self.state2
    
    def state3(self, frame):
        self.state = "power-position"
        
        midhip_y_all = self.video.midhip_y.df.values
        rwrist_y_all = self.video.rwrist_y.df.values
        
        # condition to jump to extension
        if (rwrist_y_all[frame] > midhip_y_all[frame]):
            return self.state4
        else:
            return self.state3

    def state4(self, frame):
        self.state = "extension"
        
        neck_y_all = self.video.neck_y.df.values
        rwrist_y_all = self.video.rwrist_y.df.values
        
        # condition to jump to reception (snatch / clean) (interval)
        if ((rwrist_y_all[frame] < neck_y_all[frame] + neck_y_all[frame] * 0.2) and (rwrist_y_all[frame] > neck_y_all[frame] - neck_y_all[frame] * 0.2)):
            if self.video.number < 107:
                return self.state5
            else:
                return self.state6
        else:
            return self.state4
    
    def state5(self, frame):
        self.state = "reception-snatch"
        
        return None

    def state6(self, frame):
        self.state = "reception-clean"
        
        neck_y_all = self.video.neck_y.df.values
        rwrist_y_all = self.video.rwrist_y.df.values
        
        # condition to jump to jerk
        if (rwrist_y_all[frame] > neck_y_all[frame] + neck_y_all[frame] * 0.2):
            return self.state7
        else:
            return self.state6
    
    def state7(self, frame):
        self.state = "jerk"
        
        return None

class Videos(object):

    def __init__(self, file, target):
        self.df = pd.read_csv(file, index_col=0)
        self.target = pd.read_csv(target, index_col=0)
        self.list = []

        vid_start = 0
        
        for i in range(self.df.shape[0]):
            vid_end = vid_start + self.df.iloc[i].values[0]
            # select rows from start to end of video
            video_df = self.target.iloc[vid_start:vid_end]
            self.list.append(Video(video_df, i))
            vid_start = vid_end

if __name__ == '__main__':
    videos = Videos('number_of_frames_per_video.csv', target='../data_augmentation/joint_data_3d.csv')
    v1 = videos.list[0]
    print(len(videos.list))