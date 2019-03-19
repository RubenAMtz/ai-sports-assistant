from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserListView
from kivy.clock import Clock
import cv2

class SimpleButton(Button):
    def __init__(self, **kwargs):
        super(SimpleButton, self).__init__(**kwargs)
        self.text = "Simple Button"
        self.size_hint = (None, None)
        self.height = 30
        self.width = 100
        self.bind(on_press=self._on_click)

    def _on_click(self, instance):
        pass


class FileDirectoryButton(SimpleButton):
    def __init__(self, **kwargs):
        super(FileDirectoryButton, self).__init__(**kwargs)
        self.text = "Open video"
        self.popup = Popup(title='Choose a video to estimate pose',
            content=Files(),
            size_hint=(None, None), size=(400, 400))

    def _on_click(self, instance):
        
        self.popup.open()





class PlayButton(SimpleButton):
    def __init__(self, **kwargs):
        super(PlayButton, self).__init__(**kwargs)
        self.text = "Play"

    def _on_click(self, instance):
        video_canvas = self.parent.parent.ids["video_canvas"]
        
        if video_canvas.loaded is not True:
            print("There is no video loaded yet")
            return
        
        if video_canvas.play_e is not None:
            print("Video is still playing...")
            return
        
        video_canvas.play_e = Clock.schedule_interval(video_canvas.play_video, 1/30)


class PauseButton(SimpleButton):
    def __init__(self, **kwargs):
        super(PauseButton, self).__init__(**kwargs)
        self.text = "Pause"

    def _on_click(self, instance):
        video_canvas = self.parent.parent.ids["video_canvas"]

        if video_canvas.loaded is not True:
            print("There is no video loaded yet")
            return
        
        if video_canvas.play_e is not None:
            video_canvas.play_e.cancel()
            video_canvas.play_e = None
            print("Video is paused.")


class ForwardButton(SimpleButton):
    def __init__(self, **kwargs):
        super(ForwardButton, self).__init__(**kwargs)
        self.text = ">>"

    def _on_click(self, instance):
        video_canvas = self.parent.parent.ids["video_canvas"]

        if video_canvas.loaded is not True:
            print("There is no video loaded yet")
            return

        video_canvas.inc_counter()
        video_canvas.update_texture()
        video_canvas.update_info_box()
        video_canvas.update_inference_box()


class BackwardButton(SimpleButton):
    def __init__(self, **kwargs):
        super(BackwardButton, self).__init__(**kwargs)
        self.text = "<<"

    def _on_click(self, instance):
        video_canvas = self.parent.parent.ids["video_canvas"]

        if video_canvas.loaded is not True:
            print("There is no video loaded yet")
            return
        
        video_canvas.dec_counter()
        video_canvas.update_texture()
        video_canvas.update_info_box()
        video_canvas.update_inference_box()


class Files(BoxLayout):
    def __init__(self, *args, **kwargs):
        super(Files, self).__init__(*args, **kwargs)
        self.orientation = "vertical"
        self.id = "files"
        self.selection = None
        self.fichoo = FileChooserListView()
        self.btn = SelectFileButton()
        self.add_widget(self.fichoo)
        self.add_widget(self.btn)
        




class SelectFileButton(SimpleButton):
    def __init__(self, **kwargs):
        super(SelectFileButton, self).__init__(**kwargs)
        self.text = "Open"
    
    def _on_click(self, instance):
        # root = self.parent.parent.parent.ids
        files = self.parent
        popup = self.parent.parent.parent.parent
        selection = None
        
        try:
            selection = files.fichoo.selection[0]
        except IndexError:
            print("Make sure to select a file")
        

        window_shape = self.parent.parent.parent.parent.parent.children[1]
        video_canvas = window_shape.ids['video_canvas']
        print(video_canvas)

        # security measure
        if video_canvas.load_e is not None:
            print("Video is still loading...")
            return
        
        video_canvas.reset()
        video_canvas.file = selection
        video_canvas.cap = cv2.VideoCapture(selection)
        video_canvas.load_e = Clock.schedule_interval(video_canvas.load_video, 1/60)
        # print(files.selection)
        # print(root)
        # close popup window
        popup.dismiss()
        