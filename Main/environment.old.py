import adbutils
import scrcpy

import time
import threading
import queue
import torch
import platform
import numpy as np
import cv2

from AI_Models.VAE.main import VAEWrapper
from AI_Models.EntityDetector.main import EntityDetector

class Environment():
    def __init__(self):
        self.device = self.get_device()
        #self.get_emulator_device()

        self.vae_model = VAEWrapper(self.device)
        self.entity_model = EntityDetector()

        self.last_time = time.time()

        self.reset()

    def reset(self):
        self.me = None
        self.friendly = []
        self.enemy = []
        self.gamemode = None

        self.ball_position = [-1, -1]
        self.friendly_scores = 0
        self.enemy_scores = 0

        self.friendly_safe_position = [-1, -1]
        self.enemy_safe_position = [-1, -1]
        self.friendly_safe_health = 0
        self.enemy_safe_health = 0

        self.friendly_gems = 0
        self.enemy_gems = 0

        self.brawlers_left = 0

        self.friendly_stars = 0
        self.enemy_stars = 0

        self.friendly_load = 0
        self.enemy_load = 0

        self.round = 0

        self.friendly_left = 0
        self.enemy_left = 0

        self.kills_enemy = 0
        self.kills_friendly = 0

    def handle_frame(self, frame_rgb):
        if frame_rgb is None:
            return

        frame_bgr = frame_rgb[..., ::-1].astype(np.uint8)

        entities = self.entity_model.predict(frame_bgr, frame_rgb)

        current_time = time.time()
        fps = 1 / (current_time - self.last_time)
        self.last_time = current_time


        self.visualize_entities(frame_bgr, entities, fps)


        scale_factor = 0.75
        height, width = frame_bgr.shape[:2]
        new_dimensions = (int(width * scale_factor), int(height * scale_factor))
        frame_bgr_resized = cv2.resize(frame_bgr, new_dimensions, interpolation=cv2.INTER_AREA)


        cv2.waitKey(1)
        cv2.imshow("TrophyHunter", frame_bgr_resized)

    def visualize_entities(self, frame_bgr, entities, fps):
        for entity in entities:
            label = entity['label']
            confidence = entity['confidence']
            bbox = entity['bbox']

            x1 = int(bbox[0] * frame_bgr.shape[1])
            y1 = int(bbox[1] * frame_bgr.shape[0])
            x2 = int(bbox[2] * frame_bgr.shape[1])
            y2 = int(bbox[3] * frame_bgr.shape[0])

            color = self.entity_model.args["label_colors"][label]
            color = tuple(map(int, color.split(',')))

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame_bgr, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if entity["health"] is not None:
                cv2.putText(frame_bgr, f"Health: {entity['health']}", (x2 + 5, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame_bgr, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    def stop(self):
        self.stop_event.set()
        self.display_thread.join()
        cv2.destroyAllWindows()

    def get_device(self):
        if torch.cuda.is_available():
            is_nvidia = False

            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                properties = torch.cuda.get_device_properties(i)
                if "NVIDIA" in properties.name.upper():
                    is_nvidia = True

            if(is_nvidia):
                print("Using NVIDIA GPU with CUDA.")
            else:
                print("Using AMD GPU with ROCm.")

            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Using Apple Metal Performance Shaders (MPS).")
            return torch.device("mps")
        elif platform.system() == "Windows":
            import torch_directml
            print("Using UNKNOWN GPU with DirectML")
            return torch_directml.device()
        else:
            print("No GPU detected. Using CPU.")
            return torch.device("cpu")
    #def get_emulator_device(self):
        #self.adb = adbutils.AdbClient(host="127.0.0.1", port=5037)
        #self.adb_device = self.adb.device()
        #self.scrcpy_device = scrcpy.Client(self.adb_device)