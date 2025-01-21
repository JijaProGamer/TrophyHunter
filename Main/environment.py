import adbutils
import scrcpy
import math
import copy
import time
import threading
import queue
import torch
import platform
import easyocr
import numpy as np
import cv2

from AI_Models.VAE.main import VAEWrapper
from AI_Models.EnvironmentClassifier.model import SEResNeXtFineTuner
#from AI_Models.EntityDetector.main import EntityDetector
from .output_controller import OutputController
from .environment_controller import EnvironmentController
from .utils import filter_ocr_results

class SmartNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2, clip=[-1, 1], seed=None):
        """
        Initialize the SmartNoise generator.

        Parameters:
        - size (int): The number of parameters to generate noise for.
        - mu (float): The long-term mean of the noise.
        - theta (float): The rate of mean reversion.
        - sigma (float): The volatility parameter.
        - dt (float): Time step for updates.
        - clip (int[]): clip the generated values to be within clip.
        - seed (int, optional): Seed for random number generator.
        """

        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.clip = clip
        self.state = np.ones(size) * mu
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Reset the noise state to the mean value."""
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) * self.dt
        dx += self.sigma * np.sqrt(self.dt) * self.rng.standard_normal(self.size)
        self.state += dx

        if self.clip:
            self.state = np.clip(self.state, self.clip[0], self.clip[1])

        return self.state

class Environment():
    def __init__(self, max_fps):
        self.max_fps = max_fps

        self.device = self.get_device()

        self.vae_model = VAEWrapper(self.device)
        self.environment_classifier = SEResNeXtFineTuner(num_classes=18, device=self.device)
        #self.entity_model = EntityDetector(self.device)

        self.environment_classifier.freeze_layers()
        self.environment_classifier.load()

        self.exploration_noise = SmartNoise(10, mu=0.0, theta=0.2, sigma=0.5, dt=0.1, clip=[-1, 1])

        self.get_emulator_device()
        self.output_controller = OutputController(self.scrcpy_device)
        self.environment_controller = EnvironmentController(self.scrcpy_device)
        self.emulator_resolution = [0, 0]

        self.ocr_reader = easyocr.Reader(['en'])

        self.last_time = time.time()

        self.last_frame_rgb = None
        self.last_frame_bgr = None
        self.last_entities = None
        self.fps = []


        ui_thread = threading.Thread(target=self.handle_ui)
        ui_thread.daemon = True
        ui_thread.start()

        self.reset()

        self.mode = "VAE_IMAGE_CAPTURER"

        #self.timestep = 0
        #self.step_sharm = 0

        self.selected_brawler = None#"SHELLY"
        self.chosen_brawler = False

        self.is_brawler_scrolling_mode = False
        self.brawler_scrolling_num = 0

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

    def handle_frame(self, frame_bgr):
        if frame_bgr is None:
            return

        self.emulator_resolution = [frame_bgr.shape[1], frame_bgr.shape[0]]
        self.output_controller.resolution = self.emulator_resolution
        self.environment_controller.update_positions(self.emulator_resolution)


        frame_rgb = frame_bgr[..., ::-1]
        
        environmentType = self.environment_classifier.predict(frame_rgb)
        acting = False
        

        if self.is_brawler_scrolling_mode:
            self.brawler_scrolling_num += 1
            if self.brawler_scrolling_num > 500:
                self.brawler_scrolling_num = 0
                self.is_brawler_scrolling_mode = False

            scroll_position = self.environment_controller.positions["character_scroll_position"]

            white_mask = (
                #(abs(frame_bgr[:, :, 0] - frame_bgr[:, :, 1]) < 5) &
                #(abs(frame_bgr[:, :, 1] - frame_bgr[:, :, 2]) < 5) &
                (frame_bgr[:, :, 0] > 230) &
                (frame_bgr[:, :, 1] > 230) &
                (frame_bgr[:, :, 2] > 230)
            )

            gray_roi = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) * white_mask.astype(np.uint8)

            ocr_results = self.ocr_reader.readtext(gray_roi)
            ocr_results = filter_ocr_results(ocr_results, self.environment_controller.args["brawlers"], 1)

            search = next((item for item in ocr_results if item['text'] == self.selected_brawler), None)

            if search is not None:
                self.brawler_scrolling_num = 0
                self.is_brawler_scrolling_mode = False

                self.environment_controller.press_button_raw(search["pos"][0], search["pos"][1])
                time.sleep(2)
            else:
                for i in range(1, 5):
                    self.environment_controller.move_start("character_scroll_position")
                    self.environment_controller.move(scroll_position[0], scroll_position[1] - 50)
                    time.sleep(0.02)
                    self.environment_controller.move_stop("character_scroll_position")
        else:
            if self.mode == "VAE_IMAGE_CAPTURER":
                if "Lobby" in environmentType:
                    if self.chosen_brawler:
                        # switch gamemode
                        # for now lazy way, just press play

                        self.chosen_brawler = False

                        self.environment_controller.press_button("play")
                        time.sleep(2)
                    else:
                        time.sleep(2)
                        self.environment_controller.press_button("position_character")
                        time.sleep(2)
                        self.chosen_brawler = True

                elif "Brawlers" in environmentType and self.chosen_brawler:
                    if self.selected_brawler == None:
                        self.selected_brawler = "SHELLY"
                    else:
                        index = self.environment_controller.args["brawlers"].index(self.selected_brawler)
                        if index >= len(self.environment_controller.args["brawlers"]):
                            index = -1

                        self.selected_brawler = self.environment_controller.args["brawlers"][index + 1]

                    scroll_position = self.environment_controller.positions["character_scroll_position"]

                    for i in range(1, 300):
                        self.environment_controller.move_start("character_scroll_position")
                        self.environment_controller.move(scroll_position[0], scroll_position[1] + 50)
                        time.sleep(0.05)
                        self.environment_controller.move_stop("character_scroll_position")

                    self.is_brawler_scrolling_mode = True
                    self.brawler_scrolling_num = 0
                elif "BrawlerSelector" in environmentType:
                    self.environment_controller.press_button("select_character")
                    time.sleep(2)
                elif "Game" in environmentType:
                    if not ("Wait" in environmentType) and not ("Dead" in environmentType):
                        acting = True
                        noise = self.exploration_noise.sample()
                        #noise[1] -= 0.0025
                        self.output_controller.act(noise)

        if not acting:
            self.output_controller.stop()


        self.last_frame_rgb = frame_rgb
        self.last_frame_bgr = frame_bgr
        #self.last_entities = entities

        current_time = time.time()
        self.fps.append(current_time - self.last_time)
        self.last_time = current_time

        if len(self.fps) > 20:
            del self.fps[0]
        
    def handle_ui(self):
        while True:
            cv2.waitKey(1)

            if self.last_frame_rgb is None:
                continue

            frame_bgr = self.last_frame_bgr.astype(np.uint8)
            frame_rgb = self.last_frame_rgb
            #entities = self.last_entities
            fps = 1 / np.mean(self.fps)

            if frame_bgr is None or frame_rgb is None or fps is None:
                continue

            self.last_frame_rgb = None
            self.last_frame_bgr = None
            #self.last_entities = None

            self.visualize_entities(frame_bgr, fps)

            scale_factor = 1.5
            height, width = frame_bgr.shape[:2]
            new_dimensions = (int(width * scale_factor), int(height * scale_factor))
            frame_bgr_resized = cv2.resize(frame_bgr, new_dimensions, interpolation=cv2.INTER_AREA)

            cv2.imshow("TrophyHunter", frame_bgr_resized)

    def visualize_entities(self, frame_bgr, fps):
        """for entity in entities:
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
                cv2.putText(frame_bgr, f"Health: {entity['health']}", (x2 + 5, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)"""

        cv2.putText(frame_bgr, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


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
    def get_emulator_device(self):
        self.adb = adbutils.AdbClient(host="127.0.0.1", port=5037)

        self.adb_device = self.adb.device()
        self.scrcpy_device = scrcpy.Client(self.adb_device, bitrate=(10**6 * 16), max_width=math.floor(self.vae_model.args["resolution"][0] * 8), max_fps=self.max_fps)