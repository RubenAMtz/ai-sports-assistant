# pylint: disable=no-member
import pandas as pd
import numpy as np

class Video(object):

    def __init__(self, df, avoid=False):
        self.df = pd.DataFrame(df)
        if avoid:
            return
        
        for col in self.df.columns:
            setattr(self, col, Video(self.df[col], avoid=True))

    def frame(self, f):
        return self.df.iloc[f]
    
    def __str__(self):
        return str(self.df)

class Videos(object):

    def __init__(self, file, target):
        self.df = pd.read_csv(file, index_col=0)
        self.target = pd.read_csv(target, index_col=0)
        self.list = []

        vid_start = 0
        
        for i in range(self.df.shape[0]):
            vid_end = vid_start + self.df.iloc[i].values[0]
            video_df = self.target.iloc[vid_start:vid_end]
            self.list.append(Video(video_df))
            vid_start = vid_end

if __name__ == '__main__':
    videos = Videos('number_of_frames_per_video.csv', target='../data_augmentation/joint_data_3d.csv')
    v1 = videos.list[0]
    print(len(videos.list))