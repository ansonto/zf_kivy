from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import cv2
import numpy as np
from kivy.clock import Clock
from threading import Thread

Builder.load_string('''
<HomeScreen>:
    BoxLayout:
        orientation: 'vertical'
        spacing: 10
        padding: 10
        Image:
            source: 'C:/Users/diyab/OneDrive/Documents/University/4th Year/Programming/Python/GP/EMBRYOS/LOGO2.jpeg'
            size_hint: (1, 0.3)
        Label:
            text: 'Zebrafish Detector'
            size_hint_y: None
            height: '120dp'
        Button:
            text: 'Go to Live Camera'
            size_hint_y: None
            height: '60dp'
            on_press: app.root.current = 'camera_screen'
        Button:
            text: 'Go to Upload'
            size_hint_y: None
            height: '60dp'
            on_press: app.root.current = 'upload_screen'

<CameraScreen>:
    BoxLayout:
        orientation: 'vertical'
        Button:
            text: 'Play'
            on_press: root.start_processing()
            size_hint_y: None
            height: '48dp'
        Button:
            text: 'Home'
            size_hint_y: None
            height: '48dp'
            on_press: app.root.current = 'home_screen'
        Image:
            id: live_feed
            size_hint_y: 1
            height: '300dp'
        Image:
            id: processed_display
            size_hint_y: 1
            height: '300dp'
        Label:
            id: fluorescence_label
            size_hint_y: None
            height: '48dp'
            text: 'Fluorescence level: Not Calculated'

<UploadScreen>:
    BoxLayout:
        orientation: 'vertical'
        FileChooserIconView:
            id: filechooser
            size_hint_y: 0.7
        BoxLayout:
            size_hint_y: 0.3
            orientation: 'vertical'
            Button:
                text: 'Test Image/Video'
                size_hint_y: None
                height: '48dp'
                on_press: root.test_file(filechooser.selection and filechooser.selection[0])
            Button:
                text: 'Home'
                size_hint_y: None
                height: '48dp'
                on_press: app.root.current = 'home_screen'
        Label:
            id: fluorescence_label
            size_hint_y: None
            height: '48dp'
            text: 'Fluorescence level: Not Calculated'
        Image:
            id: image_display
            size_hint_y: None
            height: '300dp'
''')

class HomeScreen(Screen):
    pass


class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super(CameraScreen, self).__init__(**kwargs)
        self.capture = None
        self.previous_frame = None

    def start_processing(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            thread = Thread(target=self.update_frames)
            thread.daemon = True
            thread.start()
        else:
            self.capture.release()
            self.capture = None

    def update_frames(self):
        while self.capture is not None and self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                motion_frame = frame.copy() 
                analysis_frame = frame.copy()  

                motion_frame = self.detect_motion(motion_frame)
                Clock.schedule_once(lambda dt: self.display_frame(motion_frame, 'live_feed'))
                
                processed_frame, fluorescence_level = self.process_frame(analysis_frame)
                Clock.schedule_once(lambda dt: self.display_frame(processed_frame, 'processed_display'))
                self.ids.fluorescence_label.text = f"Average Fluorescence Level: {fluorescence_level:.2f}"

    def detect_motion(self, frame):
        if self.previous_frame is None:
            self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame
        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(self.previous_frame, current_frame_gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500: 
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  
        self.previous_frame = current_frame_gray
        return frame

    def process_frame(self, frame):
        frame_with_contour = self.draw_zebrafish_contour(frame)
        hsv_image = cv2.cvtColor(frame_with_contour, cv2.COLOR_BGR2HSV)
        lower_hue = np.array([35, 50, 50])
        upper_hue = np.array([85, 255, 255])
        mask = cv2.inRange(hsv_image, lower_hue, upper_hue)
        fluorescent_regions = cv2.bitwise_and(frame_with_contour, frame_with_contour, mask=mask)
        gray_fluorescent = cv2.cvtColor(fluorescent_regions, cv2.COLOR_BGR2GRAY)
        fluorescent_pixels = gray_fluorescent[gray_fluorescent > 0]
        fluorescence_level = np.mean(fluorescent_pixels) if len(fluorescent_pixels) > 0 else 0
        return frame_with_contour, fluorescence_level

    def draw_zebrafish_contour(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_frame, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                cv2.drawContours(frame, [contour], -1, (255, 255, 255), 2)  
        return frame

    def display_frame(self, frame, widget_id):
        if frame is not None:
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            getattr(self.ids, widget_id).texture = texture


class UploadScreen(Screen):
    def test_file(self, path):
        if path.lower().endswith(('.png', '.jpg', '.jpeg')):
            frame = cv2.imread(path)
            self.process_and_display(frame)
        elif path.lower().endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(path)
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    self.process_and_display(frame)
            finally:
                cap.release() 

    def process_and_display(self, frame):
        if frame is not None:
            frame_with_contour = self.draw_zebrafish_contour(frame)
            processed_frame, fluorescence_level = self.process_frame(frame_with_contour)
            self.ids.image_display.texture = self.display_image(processed_frame)
            self.ids.fluorescence_label.text = f"Average Fluorescence Level: {fluorescence_level:.2f}"
        else:
            self.ids.fluorescence_label.text = "No Image Loaded or Image is Invalid"

    def draw_zebrafish_contour(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_frame, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(frame, [largest_contour], -1, (255, 255, 255), 2)
        return frame

    def process_frame(self, frame):
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hue = np.array([35, 50, 50])
        upper_hue = np.array([85, 255, 255])
        mask = cv2.inRange(hsv_image, lower_hue, upper_hue)
        fluorescent_regions = cv2.bitwise_and(frame, frame, mask=mask)
        gray_fluorescent = cv2.cvtColor(fluorescent_regions, cv2.COLOR_BGR2GRAY)
        fluorescent_pixels = gray_fluorescent[gray_fluorescent > 0]
        fluorescence_level = np.mean(fluorescent_pixels) if len(fluorescent_pixels) > 0 else 0
        return frame, fluorescence_level

    def display_image(self, frame):
        if frame is not None:
            buffer = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, bufferfmt='ubyte', colorfmt='bgr')
            return texture
        else:
            return None

class ZebrafishDetector(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(HomeScreen(name='home_screen'))
        sm.add_widget(CameraScreen(name='camera_screen'))
        sm.add_widget(UploadScreen(name='upload_screen'))
        return sm

if __name__ == '__main__':
    ZebrafishDetector().run()
