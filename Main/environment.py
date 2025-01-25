import adbutils
import scrcpy
import math
import time
import threading
import torch
import platform
import easyocr
import uuid
import numpy as np
import cv2
import random

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
        - theta (float or array-like): The rate of mean reversion, scalar or per-variable array.
        - sigma (float): The volatility parameter.
        - dt (float): Time step for updates.
        - clip (int[]): Clip the generated values to be within clip.
        - seed (int, optional): Seed for random number generator.
        """
        self.size = size
        self.mu = mu
        self.theta = np.full(size, theta) if np.isscalar(theta) else np.array(theta)
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
        self.environment_classifier = SEResNeXtFineTuner(device=self.device)
        #self.entity_model = EntityDetector(self.device)

        self.environment_classifier.freeze_layers()
        self.environment_classifier.load()

        #self.exploration_noise = SmartNoise(10, mu=0.0, theta=[0.2, 0.2, 0.5, 0.5, 0.75, 0.5, 0.5, 0.75, 0.75, 0.75], sigma=0.35, dt=0.35, clip=[-1, 1])
        self.exploration_noise = SmartNoise(10, mu=0.0, theta=0.1, sigma=0.35, dt=0.35, clip=[-1, 1])

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

        self.selected_brawler = None
        self.chosen_brawler = False
        self.brawler_scrolling_direction = None
        self.brawlers_visible = []

        self.selected_mode = None
        self.chosen_mode = False
        self.mode_scrolling_direction = None
        self.modes_visible = []

        self.last_frames = []

        self.is_brawler_scrolling_mode = False
        self.is_mode_scrolling_mode = False

        self.found_brawler = False
        self.found_mode = False

        self.selected_skin = False

        self.awaitMode = None
        self.blockMode = None

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
        self.environment_controller.update_positions(self.emulator_resolution, self.real_emulator_resolution)


        frame_rgb = frame_bgr[..., ::-1]
        
        #entities = self.entity_model.
        environmentType = self.environment_classifier.predict(frame_rgb)
        acting = False
        canDoActions = True

        if self.awaitMode is not None:
            if not (self.awaitMode in environmentType) or (self.blockMode in environmentType):
                canDoActions = False
            else:
                self.blockMode = None
                self.awaitMode = None

        if canDoActions:
            if self.is_brawler_scrolling_mode:
                move_direction = 1 if self.brawler_scrolling_direction == "Up" else -1
                move_direction *= 300

                scroll_position = self.environment_controller.positions["character_scroll_position"]

                #white_mask = (
                #    (frame_bgr[:, :, 0] > 230) &
                #    (frame_bgr[:, :, 1] > 230) &
                #    (frame_bgr[:, :, 2] > 230)
                #)

                gray_roi = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)# * white_mask.astype(np.uint8)

                ocr_results = self.ocr_reader.readtext(gray_roi)
                ocr_results = filter_ocr_results(ocr_results, self.environment_controller.args["brawlers"], 1)

                search = next((item for item in ocr_results if item['text'] == self.selected_brawler), None)

                if search is not None:
                    if self.found_brawler:
                        self.is_brawler_scrolling_mode = False

                        self.environment_controller.press_button_raw(search["pos"][0], search["pos"][1])
                        self.awaitMode = "BrawlerSelector"
                    else:
                        self.found_brawler = True
                        time.sleep(0.5)
                else:
                    current_brawlers = [item["text"] for item in ocr_results]
                    max_visible_length = 3

                    self.brawlers_visible.append(current_brawlers)
                    if len(self.brawlers_visible) > max_visible_length:
                        del self.brawlers_visible[0]

                    if sorted(self.brawlers_visible[0]) == sorted(current_brawlers) and len(self.brawlers_visible) == max_visible_length:
                        self.brawlers_visible = []
                        self.brawler_scrolling_direction = "Up" if self.brawler_scrolling_direction == "Down" else "Down"

                    self.adb_device.swipe(scroll_position[0], scroll_position[1], scroll_position[0], scroll_position[1] + move_direction, 0.15)
                    time.sleep(0.25)
            elif self.is_mode_scrolling_mode:
                scroll_position = self.environment_controller.positions["mode_scroll_position"]

                move_direction = 1 if self.mode_scrolling_direction == "Left" else -1
                move_direction *= 300

                #white_mask = (
                #    (frame_bgr[:, :, 0] > 230) &
                #    (frame_bgr[:, :, 1] > 230) &
                #    (frame_bgr[:, :, 2] > 230)
                #)

                gray_roi = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)# * white_mask.astype(np.uint8)

                ocr_results = self.ocr_reader.readtext(gray_roi)
                ocr_results = filter_ocr_results(ocr_results, self.environment_controller.args["modes"], 1)

                ranked_result = next((item for item in ocr_results if item['text'] == "RANKED"), None)
                if ranked_result:
                    ranked_center = ranked_result['pos']
                    ranked_x, ranked_y = ranked_center[0], ranked_center[1]

                    filtered_results = []

                    for result in ocr_results:
                        if result['text'] == "RANKED":
                            continue

                        center = result['pos']
                        x, y = center[0], center[1]

                        close_x = abs(x - ranked_x) < 200
                        higher_y = y > (ranked_y + 20)


                        if not (close_x and higher_y):
                            filtered_results.append(result)

                    ocr_results = filtered_results

                search = next((item for item in ocr_results if item['text'] == self.selected_mode), None)

                if search is not None:
                    if self.found_mode:
                        self.is_mode_scrolling_mode = False

                        self.environment_controller.press_button_raw(search["pos"][0], search["pos"][1])
                        self.awaitMode = "Lobby"
                    else:
                        self.found_mode = True
                        time.sleep(0.5)
                else:
                    current_modes = [item["text"] for item in ocr_results]
                    max_visible_length = 3

                    self.modes_visible.append(current_modes)
                    if len(self.modes_visible) > max_visible_length:
                        del self.modes_visible[0]

                    if sorted(self.modes_visible[0]) == sorted(current_modes) and len(self.modes_visible) == max_visible_length:
                        self.modes_visible = []
                        self.selected_mode = random.choice(self.environment_controller.args["modes"])
                        while self.selected_mode == "RANKED":
                            self.selected_mode = random.choice(self.environment_controller.args["modes"])

                        self.mode_scrolling_direction = "Right" if self.mode_scrolling_direction == "Left" else "Left"

                    self.adb_device.swipe(scroll_position[0], scroll_position[1], scroll_position[0] + move_direction, scroll_position[1], 0.15)
                    time.sleep(0.25)
            else:
                if self.mode == "VAE_IMAGE_CAPTURER":
                    if "Lobby" in environmentType:
                        if self.chosen_brawler:
                            if self.chosen_mode:
                                self.chosen_brawler = False
                                self.chosen_mode = False

                                self.environment_controller.press_button("play")
                                self.awaitMode = "Game"
                            else:
                                self.modes_visible = []

                                self.environment_controller.press_button("select_modes")
                                self.chosen_mode = True
                                self.awaitMode = "Modes"
                        else:
                            self.brawlers_visible = []

                            self.environment_controller.press_button("position_character")
                            self.chosen_brawler = True
                            self.awaitMode = "Brawlers"
                    elif "Modes" in environmentType and self.chosen_mode:
                        self.selected_mode = random.choice(self.environment_controller.args["modes"])
                        while self.selected_mode == "RANKED":
                            self.selected_mode = random.choice(self.environment_controller.args["modes"])

                        self.mode_scrolling_direction = "Right"
                        self.is_mode_scrolling_mode = True
                        self.found_mode = False
                    elif "Brawlers" in environmentType and self.chosen_brawler:
                        self.selected_brawler = random.choice(self.environment_controller.args["brawlers"])

                        self.brawler_scrolling_direction = "Up"
                        self.is_brawler_scrolling_mode = True
                        self.found_brawler = False
                    elif "BrawlerSelector" in environmentType:
                        if "SkinSelector" in environmentType:
                            x1 = int(0.03 * frame_bgr.shape[1])
                            x2 = int(0.19 * frame_bgr.shape[1])
                            y1 = int(0.83 * frame_bgr.shape[0])
                            y2 = int(0.95 * frame_bgr.shape[0])

                            roi = frame_bgr[y1:y2, x1:x2]
                            b_channel, g_channel, r_channel = cv2.split(roi)

                            mean_red = np.mean(r_channel)
                            mean_green = np.mean(g_channel)

                            if mean_red > mean_green:
                                self.environment_controller.press_button("randomise_skin")

                            self.environment_controller.press_button("exit_randomiser")
                            self.awaitMode = "BrawlerSelector"
                            self.blockMode = "SkinSelector"
                        else:
                            if self.selected_skin:
                                time.sleep(0.5)
                                self.environment_controller.press_button("select_character")
                                self.awaitMode = "Lobby"
                            else:
                                self.environment_controller.press_button("select_skin")
                                time.sleep(0.5)

                                self.awaitMode = "SkinSelector"
                                self.selected_skin = True
                    elif "Game" in environmentType:
                        if not ("Wait" in environmentType):
                            if not ("Dead" in environmentType):
                                acting = True
                                noise = self.exploration_noise.sample()
                                noise[1] -= 0.05

                                if not "HasUltra" in environmentType:
                                    noise[5] = noise[6] = noise[7] = 0

                                if not "HasGadget" in environmentType:
                                    noise[8] = 0

                                if not "HasHypercharge" in environmentType:
                                    noise[9] = 0

                                self.output_controller.act(noise)

                                small_image = cv2.resize(frame_bgr, (400, 224), cv2.INTER_AREA).astype(np.float32)
                                self.last_frames.append(small_image)


                                if len(self.last_frames) >= 60:
                                    start = time.time()
                                    average_frame = np.mean(self.last_frames, axis=0).astype(np.float32)
                                    deviations = [np.sum(np.abs(frame - average_frame)) for frame in self.last_frames]

                                    probabilities = np.array(deviations) / np.sum(deviations)

                                    selected_index = np.random.choice(len(self.last_frames), p=probabilities)
                                    most_interesting_frame = self.last_frames[selected_index]

                                    cv2.imwrite(f"VAEImages/{uuid.uuid4()}.png", most_interesting_frame)

                                    self.last_frames = []
                            else:
                                if "ShowdownExit":
                                    self.environment_controller.press_button("showdown_exit")
                    elif "Proceed":
                        self.environment_controller.press_button("proceed")
                        time.sleep(3)
                        #self.awaitMode = "Exit" # Needs to be able to handle double proceeding (for example games other than showdown)
                    elif "Exit":
                        self.environment_controller.press_button("exit")
                        self.awaitMode = "Lobby"
                    elif "MultiChose":
                        self.environment_controller.press_button("multichoose_like")
                        self.awaitMode = "Lobby"

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

            scale_factor = 1
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


        devices = self.adb.device_list()
        if not devices:
            print("No devices connected.")
            return

        print("Connected Devices:")
        for idx, device in enumerate(devices):
            print(f"[{idx}] Serial: {device.serial}")


        device = None

        if len(devices) == 1:
            device = devices[0]
        else:
            while True:
                try:
                    index = int(input("Enter the index of the device to use: "))
                    if 0 <= index < len(devices):
                        device = devices[index]
                        break
                    else:
                        print("Invalid index. Please try again.")
                except ValueError:
                    print("Please enter a valid integer.")

        if not device:
            print("No device chosen.")
            return
        

        self.adb_device = device
        self.scrcpy_device = scrcpy.Client(device, bitrate=(10**6 * 32), max_width=860, max_fps=self.max_fps)

        real_resolution = self.adb_device.shell("wm size").strip().split(": ")[1]
        real_width, real_height = map(int, real_resolution.split("x"))

        self.real_emulator_resolution = [real_width, real_height]