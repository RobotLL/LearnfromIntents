import random
import csv
import shutil
from math import radians, cos, sin
import os
import numpy as np
import pybullet as p
import pybullet_data
import itertools
import matplotlib.pyplot as plt
from generate_side_grasp import plot_grasp
import concurrent.futures
from tqdm import tqdm
from tqdm import trange
from PIL import Image 

from util import gripper_limits, objects_limits, is_point_below_line, generate_td_obj, get_depth_image
from util import normalize_01 as normalize
from find_negetive_td import get_full_action_table, get_all_good_in_root, sample_class3_from_bad

PLOT_VALID_GRASPS = False


class CommandClass:
    # INIT_STATE = 0
    # Apprached with an open gripper (z needs to be 10cm)
    # APPROACH_WITH_OPEN_GRIPPER = 1
    # Apprached with an open gripper (z needs to be 10cm)
    # Futher_APPROACH_WITH_OPEN_GRIPPER = 2
    # Go down futher and close the gripper (z need to be 5cm)
    GO_DOWN_AND_CLOSE_GRIPPER = 3

    # UNKNOW = 4


def generate_random_object_pose():

    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    table_urdf = './model/env/table.urdf'
    table_position = [0, 0, -0.05/2]
    p.loadURDF(table_urdf, table_position,
               p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

    pitch_random = 0
    roll_random = 0
    ####################################################
    #       random level
    ####################################################
    x_random = random.uniform(-1, 1) * 0.05
    y_random = random.uniform(-1, 1) * 0.05
    yaw_random = random.uniform(-1, 1) * 1.57
    present_object_id = generate_td_obj(obj_position=[
        x_random, y_random, 0.03], obj_orientation=p.getQuaternionFromEuler([pitch_random, roll_random, yaw_random]))

    for _ in range(480):
        p.stepSimulation()
    object_pos, object_orn = p.getBasePositionAndOrientation(present_object_id)
    object_orn = p.getEulerFromQuaternion(object_orn)

    depth_image = get_depth_image()

    p.disconnect()
    return object_pos, object_orn, depth_image

# depth_max = 0
# depth_min = 255
# oz_min = 1
# oz_max = 0
# for _ in trange(100):
#     object_pos, object_orn, depth_image = generate_random_object_pose()
#     depth_max = max(np.max(depth_image), depth_max)
#     depth_min = min(np.min(depth_image), depth_min)

#     oz_min = min(object_pos[2],oz_min)
#     oz_max = max(object_pos[2],oz_max)
# plt.imshow(depth_image)
# %%


def generate_top_down(object_pos, object_orn, idx):
    one_scene_data = []
    gripper_open_distance = 0.08
    object_width = 0.03
    object_length = 0.045

    observation_options = [
        # object position
        [object_pos[0]], [object_pos[1]], [object_pos[2]],
        # [0], [0], [0],
        # object orientation
        [object_orn[0]],
        [object_orn[1]],
        [object_orn[2]],

        # gripper start position
        [-0.1, -0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.1],
        [-0.1, -0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.1],
        [0],

        # gripper start orientation
        [0],
        [0],
        [-90, -60, -30, 0, 30, 60, 90],

        # gripper open
        [1]
    ]

    for index, observation in enumerate(itertools.product(*observation_options)):
        object_position = observation[0:3]
        object_orientation_euler = observation[3:6]
        gripper_position = observation[6:9]
        gripper_orientation_euler = observation[9:12]
        open_gripper = observation[12]

        object_yaw = object_orientation_euler[2]
        gripper_yaw = radians(gripper_orientation_euler[2])

        lower_edge_point = [
            object_position[0] + object_width / 2 * sin(object_yaw),
            object_position[1] - object_width / 2 * cos(object_yaw)
        ]
        upper_edge_point = [
            object_position[0] - object_width / 2 * sin(object_yaw),
            object_position[1] + object_width / 2 * cos(object_yaw)
        ]
        left_edge_point = [
            object_position[0] - object_length / 2 * cos(object_yaw),
            object_position[1] - object_length / 2 * sin(object_yaw)
        ]
        right_edge_point = [
            object_position[0] + object_length / 2 * cos(object_yaw),
            object_position[1] + object_length / 2 * sin(object_yaw)
        ]

        valid = False
        for order in [0, np.pi]:
            upper_finger_x = gripper_position[0] + 0.5 * \
                gripper_open_distance * cos(gripper_yaw + np.pi / 2 + order)
            upper_finger_y = gripper_position[1] + 0.5 * \
                gripper_open_distance * sin(gripper_yaw + np.pi / 2 + order)
            lower_finger_x = gripper_position[0] + 0.5 * gripper_open_distance * cos(
                gripper_yaw + np.pi * 3 / 2 + order)
            lower_finger_y = gripper_position[1] + 0.5 * gripper_open_distance * sin(
                gripper_yaw + np.pi * 3 / 2 + order)

            valid_sides = is_point_below_line([upper_finger_x, upper_finger_y], left_edge_point, object_yaw + np.pi/2) and \
                not is_point_below_line([upper_finger_x, upper_finger_y], right_edge_point, object_yaw + np.pi/2) and \
                not is_point_below_line([lower_finger_x, lower_finger_y], right_edge_point, object_yaw + np.pi/2) and \
                is_point_below_line([lower_finger_x, lower_finger_y],
                                    left_edge_point, object_yaw + np.pi/2)
            valid_top_down = is_point_below_line([lower_finger_x, lower_finger_y], lower_edge_point, object_yaw) and \
                not is_point_below_line(
                    [upper_finger_x, upper_finger_y], upper_edge_point, object_yaw)
            valid = valid_top_down and valid_sides
            if valid:
                break

        if valid:
            if PLOT_VALID_GRASPS:
                plot_grasp(
                    lower_finger_x,
                    lower_finger_y,
                    upper_finger_x,
                    upper_finger_y,
                    object_position,
                    object_width,
                    object_length,
                    angle=object_yaw,
                    index=index,
                    text=f'gyaw: {gripper_yaw:.1}; oyaw: {object_yaw:.1}'
                )

            one_scene_data.append([
                *object_position,
                *object_orientation_euler,
                *gripper_position,
                *gripper_orientation_euler,
                open_gripper,
                CommandClass.GO_DOWN_AND_CLOSE_GRIPPER,
                idx,
                0,
            ])

    return np.array(one_scene_data)

# if PLOT_VALID_GRASPS:
#     if os.path.exists('grasps'):
#         shutil.rmtree('grasps')
#     os.mkdir('grasps')

# object_pos, object_orn = generate_random_object_pose()
# a = generate_top_down(object_pos, object_orn)
# if len(a)==0:
#     print(object_pos, object_orn)


def collect_data(save_data_name, c_id, for_train = True):

    if PLOT_VALID_GRASPS:
        if os.path.exists('grasps'):
            shutil.rmtree('grasps')
        os.mkdir('grasps')
        
    
    if for_train == True:
        i_save_path = './td_grasp_i_train/'
        if os.path.exists(i_save_path):
            pass
        else:
            os.mkdir('td_grasp_i_train')
        
    else:
        i_save_path = './td_grasp_i_test/'
        if os.path.exists(i_save_path):
            pass
        else:
            os.mkdir('td_grasp_i_test')
    i = c_id
    object_pos, object_orn, depth_i = generate_random_object_pose()
    root_c1 = generate_top_down(object_pos, object_orn, idx=i)
    # print(len(root_c1))
    while len(root_c1) == 0:
        object_pos, object_orn, depth_i = generate_random_object_pose()
        root_c1 = generate_top_down(object_pos, object_orn, idx=i)
        
    Image.fromarray(depth_i).convert('L').save(i_save_path+str(i)+'.png')

    full_table = get_full_action_table(object_pos, object_orn)
    good_table = get_all_good_in_root(root_c1, object_pos, object_orn)
    bad_table = sample_class3_from_bad(full_table, good_table)
    data_for_one_scene = np.vstack([good_table, bad_table])
    data_for_multi_scene = data_for_one_scene

    # calculate distance between gripper and object
    distance = np.round(np.sqrt(np.square(data_for_multi_scene[:, 6]-data_for_multi_scene[:, 0])+np.square(
        data_for_multi_scene[:, 7]-data_for_multi_scene[:, 1])), decimals=3)
    distance = np.reshape(distance, [len(distance), 1])
    label = np.reshape(data_for_multi_scene[:, 13], [
                       len(data_for_multi_scene[:, 13]), 1])
    
    idx = np.reshape(data_for_multi_scene[:, 14], [
                       len(data_for_multi_scene[:, 14]), 1])
    
    data_for_multi_scene = np.hstack(
        [data_for_multi_scene[:, 0:13], distance, label, idx])

    # normalize
    data_for_multi_scene[:, 0] = normalize(
        data_for_multi_scene[:, 0], objects_limits['o_x'])
    data_for_multi_scene[:, 1] = normalize(
        data_for_multi_scene[:, 1], objects_limits['o_y'])
    data_for_multi_scene[:, 2] = normalize(
        data_for_multi_scene[:, 2], objects_limits['o_z'])
    data_for_multi_scene[:, 3] = normalize(
        data_for_multi_scene[:, 3], objects_limits['o_roll'])
    data_for_multi_scene[:, 4] = normalize(
        data_for_multi_scene[:, 4], objects_limits['o_pitch'])
    data_for_multi_scene[:, 5] = normalize(
        data_for_multi_scene[:, 5], objects_limits['o_yaw'])
    data_for_multi_scene[:, 6] = normalize(
        data_for_multi_scene[:, 6], gripper_limits['joint_x'])
    data_for_multi_scene[:, 7] = normalize(
        data_for_multi_scene[:, 7], gripper_limits['joint_y'])
    data_for_multi_scene[:, 8] = normalize(
        data_for_multi_scene[:, 8], gripper_limits['joint_z'])
    data_for_multi_scene[:, 9] = normalize(
        data_for_multi_scene[:, 9], gripper_limits['joint_roll'])
    data_for_multi_scene[:, 10] = normalize(
        data_for_multi_scene[:, 10], gripper_limits['joint_pitch'])
    data_for_multi_scene[:, 11] = normalize(
        data_for_multi_scene[:, 11], gripper_limits['joint_yaw'])
    data_for_multi_scene[:, 12] = normalize(
        data_for_multi_scene[:, 12], gripper_limits['joint_open'])
    data_for_multi_scene[:, 13] = normalize(
        data_for_multi_scene[:, 13], gripper_limits['joint_distance'])

    np.savetxt(save_data_name, data_for_multi_scene)


if __name__ == '__main__':
    temp_save_path = './temp/'
    if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)
    number_need_collect = 2000
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(collect_data(
            temp_save_path + str(floder_id)+'.csv', floder_id, for_train = True)) for floder_id in range(number_need_collect)]

    data_for_multi_scene = []
    for i in tqdm(range(number_need_collect)):
        data_for_one_scene = np.loadtxt(temp_save_path + str(i)+'.csv')
        if i == 0:
            data_for_multi_scene = data_for_one_scene
        else:
            data_for_multi_scene = np.vstack(
                [data_for_multi_scene, data_for_one_scene])
            
    i_save_path = './td_grasp_i_train/'
    img_files_path = []
    for i in range(len(data_for_multi_scene)):
        img_files_path.append(i_save_path+str(int(data_for_multi_scene[i, 15]))+'.png')
    img_files_path = np.array(img_files_path).reshape(-1,1)
    
    np.save('train_td_data.npy',np.hstack([data_for_multi_scene[:, :15],img_files_path]))
    
    shutil.rmtree(temp_save_path)

    temp_save_path = './temp/'
    if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)
    number_need_collect = 200
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(collect_data(
            temp_save_path + str(floder_id)+'.csv', floder_id, for_train = False)) for floder_id in range(number_need_collect)]

    data_for_multi_scene = []
    for i in tqdm(range(number_need_collect)):
        data_for_one_scene = np.loadtxt(temp_save_path + str(i)+'.csv')
        if i == 0:
            data_for_multi_scene = data_for_one_scene
        else:
            data_for_multi_scene = np.vstack(
                [data_for_multi_scene, data_for_one_scene])
            
    i_save_path = './td_grasp_i_test/'
    img_files_path = []
    for i in range(len(data_for_multi_scene)):
        img_files_path.append(i_save_path+str(int(data_for_multi_scene[i, 15]))+'.png')
    img_files_path = np.array(img_files_path).reshape(-1,1)
    
    np.save('test_td_data.npy',np.hstack([data_for_multi_scene[:, :15],img_files_path]))
    
    shutil.rmtree(temp_save_path)
