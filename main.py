import os
import time


import time

from Main.recorder import Recorder
from Main.output_controller import OutputController
from Main.environment import Environment

directory = './random'

for item in os.listdir(directory):
    item_path = os.path.join(directory, item)
    if os.path.isfile(item_path) or os.path.islink(item_path):
        os.unlink(item_path) 
    elif os.path.isdir(item_path):
        os.rmdir(item_path)


#app_name = "BlueStacks"
fps = 100

environment = Environment()

recorder = Recorder(environment.scrcpy_device, fps, environment.handle_frame)

#recorder = Recorder(app_name, fps, environment.handle_frame)
#output_controller = OutputController(recorder.bbox)

"""import math

timestep = 0
step_sharm = 0

start = time.time()
while True:
    timestep += 0.06
    step_sharm += 1

    x = math.cos(timestep)
    y = math.sin(timestep)
    shoot_x = -x
    shoot_y = -y
    shoot = int(step_sharm % 200 == 0)
    if step_sharm % 100 == 0 and not shoot:
        shoot_x = 0
        shoot_y = 0


    #output_controller.act([x, y, shoot_x, shoot_y, shoot, 0, 0, 0, 0, 0])
    output_controller.act([x, y, 0, 0, 0, shoot_x, shoot_y, shoot, 1, 0])

    time.sleep(0.03)

    if time.time() - start > 10:
        break"""

recorder.start()
#cv2.destroyAllWindows()