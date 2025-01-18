import threading
import pygetwindow as gw
import numpy as np
import time
import cv2
import dxcam

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
        self.small_bbox = (bbox["left"], bbox["top"], bbox["left"] + bbox["width"], bbox["top"] + bbox["height"])
        self.handle_frame = handle_frame
        self.stopped = False
        self.cam = dxcam.create()

        self.frame_buffer = None
        self.lock = threading.Lock()

    def capture_frames(self):
        top_bar = 32
        self.cam.start()
        while not self.stopped:
            frame = self.cam.grab(self.small_bbox)
            if frame is None or len(frame.shape) == 0:
                continue

            img = np.array(frame)[top_bar:, :, :3]

            with self.lock:
                self.frame_buffer = img

        self.cam.stop()

    def process_frames(self):
        last_fire_time = time.time()
        while not self.stopped:
            #if time.time() - last_fire_time < self.interval:
            #    continue

            with self.lock:
                if self.frame_buffer is not None:
                    frame_to_process = self.frame_buffer.copy()
                else:
                    continue

            if frame_to_process is not None:
                last_fire_time = time.time()
                self.handle_frame(frame_to_process)
            


    def start(self):
        self.stopped = False

        capture_thread = threading.Thread(target=self.capture_frames)
        process_thread = threading.Thread(target=self.process_frames)

        capture_thread.start()
        process_thread.start()

        capture_thread.join()
        process_thread.join()

    def stop(self):
        self.stopped = True
