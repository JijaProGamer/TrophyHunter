import os
import time
import yaml
import threading
import scrcpy

class EnvironmentController():
    def __init__(self, scrcpy_device):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        with open(os.path.join(dir_path, 'environment_constants.yaml')) as f:
            self.args = yaml.safe_load(f)


        self.scrcpy_device = scrcpy_device

    def update_positions(self, resolution):
        self.resolution = resolution
        self.positions = self.calculate_positions(resolution)

    def calculate_positions(self, resolution):
        return {
            "position_character": [self.args["position_character"][0] * resolution[0], self.args["position_character"][1] * resolution[1]],
            "character_scroll_position": [self.args["character_scroll_position"][0] * resolution[0], self.args["character_scroll_position"][1] * resolution[1]],
            "select_character": [self.args["select_character"][0] * resolution[0], self.args["select_character"][1] * resolution[1]],
            "play": [self.args["play"][0] * resolution[0], self.args["play"][1] * resolution[1]],
        }

    def press_button(self, name):
        position = self.positions[name]

        self.scrcpy_device.control.touch(position[0], position[1], scrcpy.ACTION_DOWN, 0)
        time.sleep(0.1)
        self.scrcpy_device.control.touch(position[0], position[1], scrcpy.ACTION_UP, 0)
    
    def press_button_raw(self, x, y):
        self.scrcpy_device.control.touch(x, y, scrcpy.ACTION_DOWN, 0)
        time.sleep(0.1)
        self.scrcpy_device.control.touch(x, y, scrcpy.ACTION_UP, 0)



    def move_start(self, name):
        position = self.positions[name]

        self.scrcpy_device.control.touch(position[0], position[1], scrcpy.ACTION_DOWN, 0)

    def move_stop(self, name):
        position = self.positions[name]

        self.scrcpy_device.control.touch(position[0], position[1], scrcpy.ACTION_DOWN, 0)

    def move(self, x, y):
        self.scrcpy_device.control.touch(x, y, scrcpy.ACTION_MOVE, 0)