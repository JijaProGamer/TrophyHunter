import math
import time
import threading
import scrcpy

class OutputController():
    actions_shape = {
        "shape": 10, # move_x, move_y, attack_x, attack_y, attack_release, ultra_x, ultra_y, ultra_release, gadget, hypercharge
        "min": [-1, -1, -1, -1, 0, -1, -1, 0, 0, 0],
        "max": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }

    mouse_initial_positions = {
        #"move": [0.1, 0.87, 0.05],
        #"shoot": [0.9, 0.78, 0.05],
        #"ultra": [0.82, 0.85, 0.05],
        "move": [0.1, 0.87, 0.031],
        "shoot": [0.9, 0.78, 0.031],
        "ultra": [0.82, 0.85, 0.031],
        "gadget": [0.85, 0.92],
        "hypercharge": [0.78, 0.92],
    }

    def __init__(self, scrcpy_device):
        self.scrcpy_device = scrcpy_device

        self.isAiming = {"shoot": False, "ultra": False, "move": False}
        self.isSleeping = {"shoot": False, "ultra": False}
        self.sleepingTimes = {"shoot": 0, "ultra": 0}
        self.resolution = None

    def act(self, raw_actions):
        move_position = (raw_actions[0], raw_actions[1])
        attack_position = [raw_actions[2], -raw_actions[3], raw_actions[4] >= 0.5]
        ultra_position = [raw_actions[5], -raw_actions[6], raw_actions[7] >= 0.5]
        gadget = raw_actions[8] >= 0.5
        hypercharge = raw_actions[9] >= 0.5


        self.beginning_positions = self.calculate_beggining_positions()

        self.handle_walking_event(move_position, 1000)
        self.handle_attack_events(attack_position, "shoot", 1001)
        self.handle_attack_events(ultra_position, "ultra", 1002)

        if gadget:
            self.press_button(self.beginning_positions["gadget"], 1003)

        if hypercharge:
            self.press_button(self.beginning_positions["hypercharge"], 1004)

    def calculate_beggining_positions(self):
        return {
            "move": [self.mouse_initial_positions["move"][0] * self.resolution[0], self.mouse_initial_positions["move"][1] * self.resolution[1], self.mouse_initial_positions["move"][2] * self.resolution[0]],
            "shoot": [self.mouse_initial_positions["shoot"][0] * self.resolution[0], self.mouse_initial_positions["shoot"][1] * self.resolution[1], self.mouse_initial_positions["move"][2] * self.resolution[0]],
            "ultra": [self.mouse_initial_positions["ultra"][0] * self.resolution[0], self.mouse_initial_positions["ultra"][1] * self.resolution[1], self.mouse_initial_positions["move"][2] * self.resolution[0]],
            "gadget": [self.mouse_initial_positions["gadget"][0] * self.resolution[0], self.mouse_initial_positions["gadget"][1] * self.resolution[1], self.mouse_initial_positions["move"][2] * self.resolution[0]],
            "hypercharge": [self.mouse_initial_positions["hypercharge"][0] * self.resolution[0], self.mouse_initial_positions["hypercharge"][1] * self.resolution[1], self.mouse_initial_positions["move"][2] * self.resolution[0]],
        }

    def handle_walking_event(self, position, indice):
        magnitude = math.sqrt(position[0] ** 2 + position[1] ** 2)
        position_using = self.beginning_positions["move"]
           
        if magnitude < 0.2:
            if self.isAiming["move"]:
                self.scrcpy_device.control.touch(position_using[0], position_using[1], scrcpy.ACTION_UP , indice)

                self.isAiming["move"] = False
        else:
            if not self.isAiming["move"]:
                self.scrcpy_device.control.touch(position_using[0], position_using[1], scrcpy.ACTION_DOWN, indice)

                self.isAiming["move"] = True
                return

            self.scrcpy_device.control.touch(position_using[0] + position[0] * position_using[2], position_using[1] + position[1] * position_using[2], scrcpy.ACTION_MOVE, indice)

    def handle_attack_events(self, position, method_using, indice):
        position_using = self.beginning_positions[method_using]

        if self.isSleeping[method_using]:
            if time.time() - self.sleepingTimes > 0.1: 
                self.scrcpy_device.control.touch(position_using[0], position_using[1], scrcpy.ACTION_UP, indice)
                self.isSleeping[method_using] = False
            return
        
        magnitude = math.sqrt(position[0] ** 2 + position[1] ** 2)

        if position[2]:
            if self.isAiming[method_using]:
                self.scrcpy_device.control.touch(position_using[0], position_using[1], scrcpy.ACTION_MOVE, indice)
                self.scrcpy_device.control.touch(position_using[0], position_using[1], scrcpy.ACTION_UP, indice)

                self.isAiming[method_using] = False
        else: 
            if magnitude < 0.2:
                if self.isAiming[method_using]:
                    self.scrcpy_device.control.touch(position_using[0] + position[0] * position_using[2], position_using[1] + position[1] * position_using[2], scrcpy.ACTION_MOVE, indice)

                    self.isSleeping[method_using] = True
                    self.sleepingTimes = time.time()
                    self.isAiming[method_using] = False
            else:
                if not self.isAiming[method_using]:
                    self.scrcpy_device.control.touch(position_using[0], position_using[1], scrcpy.ACTION_DOWN, indice)

                    self.isAiming[method_using] = True
                    return

                self.scrcpy_device.control.touch(position_using[0] + position[0] * position_using[2], position_using[1] + position[1] * position_using[2], scrcpy.ACTION_MOVE, indice)

    def press_button(self, position, indice):
        self.scrcpy_device.control.touch(position[0], position[1], scrcpy.ACTION_DOWN, indice)
        self.scrcpy_device.control.touch(position[0], position[1], scrcpy.ACTION_UP, indice)