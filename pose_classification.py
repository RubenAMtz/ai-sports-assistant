from kivy.config import Config
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1366') # 1280
Config.set('graphics', 'height', '768') # 720


import cv2
import numpy as np

from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics.texture import Texture

from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.image import Image
from buttons import SimpleButton, FileDirectoryButton
from kivy.core.window import Window
from openpose_inference import run_openpose
from data_processing import data_loading, data_distances, normalize_distance_data
from inference import pose_inference, grade_inference

class WindowShape(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_key_down=self._on_keyboard_down)
        print(self.ids)
    
    def _on_keyboard_down(self, instance, keyboard, keycode, text, modifiers):
        backward_btn = self.ids['backward_button']
        forward_btn = self.ids['forward_button']
        
        if keycode == 80: # left key:
            backward_btn._on_click(self)
        elif keycode == 79: # right key
            forward_btn._on_click(self)

class VideoCanvas(Image):
    def __init__(self, **kwargs):
        super(VideoCanvas, self).__init__(**kwargs)
        self.texture = self.vid2texture(cv2.imread('img/default.jpg'))
        self.cap = None
        self.file = None
        self.frames = []
        # self.processed_frames = []
        self.bodykp = np.empty(0)
        self.counter = 0
        self.joint_data = None
        self.data_distances = None
        self.loaded = False
        self.pose_inference = None
        self.normalized_distances = None
        self.grade_inference = None

        # events
        self.load_e = None
        self.play_e = None

    def reset(self):
        """ Set initial values for loading videos
        """
        self.frames = []
        self.bodykp = np.empty(0)
        self.joint_data = None
        self.counter = 0
        self.cap = None
        self.file = None
        self.loaded = False
        self.data_distances = None
        self.normalized_distances = None
        self.pose_inference = None
        self.grade_inference = None
        

    def load_video(self, dt):
        """ Load video into frames variable
        """
        if self.cap is not None:
            ret, image = self.cap.read()

            if ret:
                self.frames.append(image)
                self.bodykp = np.append(self.bodykp, run_openpose(image)[0])
                print("appending frames...")
            else:
                # cancel loading video event
                self.load_e.cancel()
                self.load_e = None
                # Load first frame of video
                self.texture = self.vid2texture(self.frames[0])
                self.loaded = True
                self.update_info_box()
                
                self.bodykp = self.bodykp.reshape(-1, 3)
                self.pipeline()
                # print(self.inference)
                print("Finished loading frames.")

    def pipeline(self):
        self.joint_data = data_loading(self.bodykp, ['lear_x', 'lear_y'])
        self.data_distances = data_distances(self.joint_data)
        self.normalized_distances = normalize_distance_data(self.data_distances)
        self.pose_inference = pose_inference(self.normalized_distances)
        self.grade_inference = grade_inference(self.normalized_distances)

    def play_video(self, dt):
        """ Plays video
        """
        print("Video is playing...")
        self.update_texture()
        self.update_info_box()
        self.update_inference_box()
        self.inc_counter()
        
    
    def update_info_box(self):
        current_frame_lbl = self.parent.ids['current_frame']
        current_frame_lbl.text = str(self.counter) + " / " + str(len(self.frames) - 1)
        video_name_lbl = self.parent.ids['video_name']
        video_name_lbl.text = self.file.split("\\")[-1]
    
    def update_inference_box(self):
        pose_lbl = self.parent.ids['pose_lbl']
        pose_lbl.text = self.pose_inference[self.counter]
        grade_lbl = self.parent.ids['grade_lbl']
        grade_lbl.text = str(self.grade_inference[self.counter])
        # pose_lbl.text = self.inference[self.counter]
    
    def update_texture(self):
        """ Update current frame to show in window
        """
        self.texture = self.vid2texture(self.frames[self.counter])

    def inc_counter(self):
        """Safely increment video counter
        """
        if self.counter < len(self.frames) - 1:
            self.counter += 1
        else:
            print("Play stopped.")
            if self.play_e is not None:
                self.play_e.cancel()
            self.play_e = None
    
    def dec_counter(self):
        """ Safely decremente video counter
        """
        if self.counter > 0:
            self.counter -= 1





    def vid2texture(self, frame):
        """ Convert cv2 frame to texture object
        """
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        image_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]),
            colorfmt='bgr'
        )
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return image_texture

class PoseClassificationApp(App):
    def build(self):
        return WindowShape()


if __name__ == '__main__':
    PoseClassificationApp().run()