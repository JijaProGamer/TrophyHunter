import pyautogui
import math
import time

class OutputController():
    actions_shape = {
        "shape": 10, # move_x, move_y, attack_x, attack_y, attack_release, ultra_x, ultra_y, ultra_release, gadget, hypercharge
        "min": [-1, -1, -1, -1, 0, -1, -1, 0, 0, 0],
        "max": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }

    keycodes = {
        "gadget": "f",
        "hypercharge": "z",
        "move": {
            (0, 1): ["w"],
            (0, -1): ["s"],
            (1, 0): ["d"],
            (-1, 0): ["a"],
            (1, 1): ["w", "d"],
            (-1, 1): ["w", "a"],
            (1, -1): ["s", "d"],
            (-1, -1): ["s", "a"],
        }
    }

    mouse_initial_positions = {
        "shoot": [0.9, 0.78, 0.05],
        "ultra": [0.82, 0.84, 0.05],
    }

    def __init__(self, bbox):
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        pyautogui.DARWIN_CATCH_UP_TIME = 0

        self.bbox = bbox
        self.last_keys = []

        shoot_norm_pos = self.mouse_initial_positions["shoot"]
        ultra_norm_pos = self.mouse_initial_positions["ultra"]
        self.isAiming = False
        self.aimType = None
        self.mouse_positions = {
            "shoot": [bbox["left"] + shoot_norm_pos[0] * bbox["width"], bbox["top"] + shoot_norm_pos[1] * bbox["height"], shoot_norm_pos[2] * bbox["width"]],
            "ultra": [bbox["left"] + ultra_norm_pos[0] * bbox["width"], bbox["top"] + ultra_norm_pos[1] * bbox["height"], ultra_norm_pos[2] * bbox["width"]]
        }

    def act(self, raw_actions):
        move_position = (raw_actions[0], raw_actions[1])
        attack_position = [raw_actions[2], -raw_actions[3], raw_actions[4] >= 0.5]
        ultra_position = [raw_actions[5], -raw_actions[6], raw_actions[7] >= 0.5]
        gadget = raw_actions[8] >= 0.5
        hypercharge = raw_actions[9] >= 0.5


        magnitude_attack = math.sqrt(attack_position[0] ** 2 + attack_position[1] ** 2)
        magnitude_ultra = math.sqrt(ultra_position[0] ** 2 + ultra_position[1] ** 2)

        magnitude_types = {
            "shoot": magnitude_attack,
            "ultra": magnitude_ultra
        }

        position_types = {
            "shoot": attack_position,
            "ultra": ultra_position
        }

        if not self.isAiming:
            if magnitude_attack > 0.2:
                self.handle_mouse_events(attack_position, "shoot", magnitude_attack)
            else:
                if magnitude_ultra > 0.2:
                    self.handle_mouse_events(ultra_position, "ultra", magnitude_ultra)
        else:
            self.handle_mouse_events(position_types[self.aimType], self.aimType, magnitude_types[self.aimType])


        move_keys = self.transform_move_position(move_position)
        self.set_move_keys(move_keys)
        
        if gadget:
            pyautogui.press(self.keycodes["gadget"])

        if hypercharge:
            pyautogui.press(self.keycodes["hypercharge"])

    def handle_mouse_events(self, position, method_using, magnitude):
        position_using = self.mouse_positions[method_using]

        if position[2]:
            if self.isAiming:
                pyautogui.moveTo(position_using[0] + position[0] * position_using[2], position_using[1]+ position[1] * position_using[2], 0.01)
                pyautogui.mouseUp()
                
                self.isAiming = False
                self.aimType = None
        else:            
            if magnitude < 0.2:
                pyautogui.moveTo(position_using[0], position_using[1], 0.01)
                time.sleep(0.1)
                pyautogui.mouseUp()
                time.sleep(0.1)

                self.isAiming = False
                self.aimType = None
            else:
                if not self.isAiming:
                    pyautogui.moveTo(position_using[0], position_using[1], 0.01)
                    time.sleep(0.1)
                    pyautogui.mouseDown()
                    time.sleep(0.1)

                pyautogui.moveTo(position_using[0] + position[0] * position_using[2], position_using[1] + position[1] * position_using[2], 0.01)

                self.isAiming = True
                self.aimType = method_using
        
    def set_move_keys(self, move_keys):
        for key in move_keys:
            if not key in self.last_keys:
                pyautogui.keyDown(key)

        for old_key in self.last_keys:
            if not old_key in move_keys:
                pyautogui.keyUp(old_key)

        self.last_keys = move_keys

    def transform_move_position(self, move_position):
        magnitude = math.sqrt(move_position[0] ** 2 + move_position[1] ** 2)
        
        if magnitude <= 0.2:
            return []
        
        closest_move = None
        closest_distance = float('inf')
        for key in self.keycodes["move"]:
            distance = math.sqrt((move_position[0] - key[0]) ** 2 + (move_position[1] - key[1]) ** 2)
            if distance < closest_distance:
                closest_distance = distance
                closest_move = key
        
        return self.keycodes["move"].get(closest_move, None)