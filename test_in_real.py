# %% init and load model, init robot
import time
import stable_baselines3
import math
import matplotlib.pyplot as plt
import numpy as np
from rl_env import Sim
from real_L515 import L515
from real_robot import RealRobot
import os
from PIL import Image
from util import next_path

os.getcwd()
os.chdir('/home/rl/Documents/GitHub/RL_NLP')
classification_model = None
env = Sim(gui=False, classification_model=classification_model)
model = stable_baselines3.PPO.load('model.zip', env=env)
print('model load')
rob = RealRobot()
rob.open_gripper(100)
# %% get_depth and action
rob.set_tcp(0, 0, 0, 0, 0, 0)
rob.go_c_home()
rob.set_tcp(0, 0, 0.255+0.02, 0, 0, 0)
rob.open_gripper(100)
print('robot have go home')

l515 = L515()
verts = l515.get_verts()
full_dpi_verts = l515.get_full_dpi_verts()
l515.stop_streaming()
print('have got depth')

cz = verts[:, :, 2]

cz[np.where(cz > 0.41)] = 1
cp_w = 80
cp_h = 64
cz = cz[cp_h - 30: cp_h + 30, cp_w - 30: cp_w + 30]
plt.imshow(cz)
min_in_d = 0.3
max_in_d = 0.41

process_depth = cz.copy()
process_depth = (process_depth-min_in_d)/(max_in_d-min_in_d)*255
process_depth[np.where(cz > 0.5)] = 255

process_depth = process_depth[np.newaxis, :, :]
process_depth = np.array(process_depth).astype(np.uint8)
print('depth processed')
# %%
action_lists = []
vector = env.get_real_vector(0, 0, 0.1, 0, 0, 0, 0, 0)
for step in range(1, 4):
    obs = {'vector': vector, 'image': process_depth}
    action, _ = model.predict(obs, deterministic=True)

    new_x = (action[0]*2-10)*0.01
    new_y = (action[1]*2-10)*0.01
    new_z = action[2]*0.05
    roll = action[3]*30-30
    pitch = action[4]*45-45
    yaw = action[5]*30-90
    if_open = action[6]
    action_lists.append([new_x, new_y, new_z, roll, pitch, yaw, if_open])
    vector = env.get_real_vector(new_x, new_y, new_z, roll, pitch, yaw, if_open, step)

print('action processed')
print(action_lists)

rob.go_c_home()
rob.set_tcp(0, 0, 0.255+0.02, 0, 0, 0)
rob.open_gripper(100)
# step1:
# x,y in differet order in real
wy = action_lists[0][0]*-1
wx = action_lists[0][1]
wz = action_lists[0][2]
roll = action_lists[0][3]
pitch = action_lists[0][4]
yaw = action_lists[0][5]
print('step1:', wx, wy, wz, roll, pitch, yaw)

rob.set_gripper_ori(math.radians(roll), math.radians(pitch),  math.radians(yaw), acc=0.1, vel=0.1, wait=True)
rob.move_with_ori([wx, wy, wz+0.033+0.03-0.0015], acc=0.1, vel=0.1)
# step2:
wy = action_lists[1][0]*-1
wx = action_lists[1][1]
wz = action_lists[1][2]
roll = action_lists[1][3]-action_lists[0][3]
pitch = action_lists[1][4]-action_lists[0][4]
yaw = action_lists[1][5]-action_lists[0][5]
print('step2:', wx, wy, wz, roll, pitch, yaw)
rob.set_gripper_ori(math.radians(roll), math.radians(pitch),  math.radians(yaw))
rob.move_with_ori([wx, wy, wz+0.033+0.03-0.0015], acc=0.1, vel=0.1)
# step3:
wy = action_lists[2][0]*-1
wx = action_lists[2][1]
wz = action_lists[2][2]
roll = action_lists[2][3] - action_lists[1][3]
pitch = action_lists[2][4] - action_lists[1][4]
yaw = action_lists[2][5] - action_lists[1][5]
print('step3:', wx, wy, wz, roll, pitch, yaw)
rob.set_gripper_ori(math.radians(roll), math.radians(pitch),  math.radians(yaw))
rob.move_with_ori([wx, wy, wz+0.033+0.01], acc=0.005, vel=0.005)
# close
rob.close_gripper(200)
time.sleep(3)
rb = rob.get_rob()
for i in range(10):
    rb.movel((0, 0, 0.005, 0, 0, 0), acc=0.05, vel=0.5, relative=True, wait=True)
    rob.close_gripper(255)

# go trash and go home
rob.close_gripper(255)
rob.move_joint([64.74, -62, -127, -80, 89, -23], acc=0.3, vel=0.3)
rob.close_gripper(255)
rob.close_gripper(100)
rob.go_c_home()
