import mss
import pygetwindow as gw
import numpy as np
import time
import cv2

class Recorder():
    def __init__(self, app_name, fps, handle_frame):
        window = gw.getWindowsWithTitle(app_name)[0]

        if not window:
            print(f"Unable to find {app_name}...")
            return
        

        bbox = {
            "left": window.left,
            "top": window.top,
            "width": window.size[0],
            "height": window.size[1]
        }

        if bbox["left"] < 0 or bbox["top"] < 0 or bbox["width"] < 0 or bbox["height"] < 0:
            print(f"Unable to record, {app_name} needs to be active and on top")
            return
        
        self.interval = 1 / fps
        self.fps = fps
        self.app_name = app_name
        self.bbox = bbox
        self.handle_frame = handle_frame
        self.stopped = False
    def start(self):
        with mss.mss() as sct:
            last_fire_time = time.time()

            while True:
                frame = sct.grab(self.bbox)

                img = np.array(frame)[:, :, :3]

                if time.time() - last_fire_time >= self.interval:
                    last_fire_time = time.time()
                    self.handle_frame(img)

                if self.stopped:
                    break

                cv2.waitKey(1)
    def stop(self):
        self.stopped = True