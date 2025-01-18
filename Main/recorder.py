import scrcpy
import numpy as np
import time
import cv2
import threading

class Recorder:
    def __init__(self, scrcpy_device, fps, handle_frame):
        self.interval = 1 / fps
        self.fps = fps
        self.stopped = False
        
        self.handle_frame = handle_frame
        self.scrcpy_device = scrcpy_device

        self.is_processing = False
        self.current_frame = None
        self.processing_thread = None

    def handle_scrcpy_frame(self, frame):
        self.current_frame = frame 

    def process_frames(self):
        while not self.stopped:
            if self.current_frame is not None and not self.is_processing:
                self.is_processing = True
                self.handle_frame(self.current_frame)
                self.is_processing = False
            time.sleep(0.001)

    def start(self):
        self.scrcpy_device.add_listener(scrcpy.EVENT_FRAME, self.handle_scrcpy_frame)
        self.scrcpy_device.start(threaded=True)

        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def stop(self):
        self.stopped = True
            
        if self.processing_thread:
            self.processing_thread.join()

        self.scrcpy_device.remove_listener(scrcpy.EVENT_FRAME)
        self.scrcpy_device.stop()