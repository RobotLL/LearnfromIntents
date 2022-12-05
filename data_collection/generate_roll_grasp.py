import random
import shutil
from math import radians, cos, sin, sqrt, atan2
import os
import numpy as np
import pybullet as p
import itertools
import concurrent.futures
from tqdm import tqdm, trange
from generate_side_grasp import plot_grasp
import matplotlib.pyplot as plt
from PIL import Image 

from util import is_point_below_line, is_point_below_two_point_line, does_line_intersects_circle
from util import gripper_limits, objects_limits, generate_roll_obj, get_depth_image
from util import normalize_01 as normalize
from find_negetive_roll import get_full_action_table, get_all_good_in_root, sample_class3_from_bad


PLOT_VALID_GRASPS = False


class CommandClass:
    ROLL_GRASP = 3
    # ROLL_Futher = 3


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
    present_object_id = generate_roll_obj(obj_position=[
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
# for _ in trange(1):
#     object_pos, object_orn, depth_image = generate_random_object_pose()
#     depth_max = max(np.max(depth_image), depth_max)
#     depth_min = min(np.min(depth_image), depth_min)

#     oz_min = min(object_pos[2],oz_min)
#     oz_max = max(object_pos[2],oz_max)
# plt.imshow(depth_image)
# %%


def generate_roll(object_pos, object_orn, idx):
    one_scene_data = []
    object_radius = 0.031
    gripper_open_distance = 0.08
    max_overshoot = 0

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
        [0.05],

        # gripper start orientation
        [-45, 45],
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

        # Gripper can grasp the object perfectly if the angle distance between object and gripper is 0.
        object_yaw = object_orientation_euler[2]
        gripper_yaw = radians(gripper_orientation_euler[2])
        gripper_roll = radians(gripper_orientation_euler[0])

        # Calculate finger positions
        roll_compensation = np.pi if gripper_roll > 0 else 0
        upper_finger_x = gripper_position[0] + 0.5 * gripper_open_distance * cos(
            gripper_yaw + np.pi / 2 + roll_compensation)
        upper_finger_y = gripper_position[1] + 0.5 * gripper_open_distance * sin(
            gripper_yaw + np.pi / 2 + roll_compensation)
        lower_finger_x = gripper_position[0] + 0.5 * gripper_open_distance * cos(
            gripper_yaw + np.pi * 3 / 2 + roll_compensation)
        lower_finger_y = gripper_position[1] + 0.5 * gripper_open_distance * sin(
            gripper_yaw + np.pi * 3 / 2 + roll_compensation)

        is_in_front_of_circle = (is_point_below_two_point_line(object_position[:2], [
            lower_finger_x, lower_finger_y], [upper_finger_x, upper_finger_y]))
        is_between_fingers_1 = (is_point_below_line(object_position[:2], [upper_finger_x, upper_finger_y], gripper_yaw) and not is_point_below_line(
            object_position[:2], [lower_finger_x, lower_finger_y], gripper_yaw)) and gripper_roll < 0
        is_between_fingers_2 = (not is_point_below_line(object_position[:2], [upper_finger_x, upper_finger_y], gripper_yaw) and is_point_below_line(
            object_position[:2], [lower_finger_x, lower_finger_y], gripper_yaw)) and gripper_roll > 0
        is_gripper_center_close_to_circle = ((gripper_position[0] - object_position[0]) ** 2 + (
            gripper_position[1] - object_position[1]) ** 2 <= (object_radius + max_overshoot) ** 2)
        are_fingers_touching_circle = does_line_intersects_circle([lower_finger_x, lower_finger_y], gripper_yaw, object_position[:2], object_radius) or does_line_intersects_circle([
            upper_finger_x, upper_finger_y], gripper_yaw, object_position[:2], object_radius)

        # Check whether the grasp is valid
        valid = is_in_front_of_circle and (
            is_between_fingers_1 or is_between_fingers_2) and is_gripper_center_close_to_circle and not are_fingers_touching_circle
        if valid:
            if PLOT_VALID_GRASPS:
                plot_grasp(
                    upper_finger_x,
                    upper_finger_y,
                    lower_finger_x,
                    lower_finger_y,
                    object_position,
                    object_radius,
                    0,
                    angle=object_yaw,
                    index=index,
                    text=f'{gripper_yaw:.2} {gripper_roll:.2}',
                    extra_points=[
                        gripper_position[:2]
                    ],
                )

            one_scene_data.append([
                *object_position,
                *object_orientation_euler,
                *gripper_position,
                *gripper_orientation_euler,
                open_gripper,
                CommandClass.ROLL_GRASP,
                idx,
                0,
            ])

    # todo: temp fix, + - roll problem
    one_scene_data = np.array(one_scene_data)
    one_scene_data[:, 9] = one_scene_data[:, 9]*-1
    return np.array(one_scene_data)


def collect_data(save_data_name, c_id, for_train = True):

    if PLOT_VALID_GRASPS:
        if os.path.exists('grasps'):
            shutil.rmtree('grasps')
        os.mkdir('grasps')

    
    if for_train == True:
        i_save_path = './roll_grasp_i_train/'
        if os.path.exists(i_save_path):
            pass
        else:
            os.mkdir('roll_grasp_i_train')
        
    else:
        i_save_path = './roll_grasp_i_test/'
        if os.path.exists(i_save_path):
            pass
        else:
            os.mkdir('roll_grasp_i_test')

    i = c_id
    # print('collect_number: ', i)
    object_pos, object_orn, depth_i = generate_random_object_pose()
    root_c1 = generate_roll(object_pos, object_orn, idx=i)
    # print(len(root_c1))
    while len(root_c1) == 0:
        object_pos, object_orn, depth_i = generate_random_object_pose()
        root_c1 = generate_roll(object_pos, object_orn, idx=i)
        
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
            
    i_save_path = './roll_grasp_i_train/'
    img_files_path = []
    for i in range(len(data_for_multi_scene)):
        img_files_path.append(i_save_path+str(int(data_for_multi_scene[i, 15]))+'.png')
    img_files_path = np.array(img_files_path).reshape(-1,1)
    
    np.save('train_roll_data.npy',np.hstack([data_for_multi_scene[:, :15],img_files_path]))
    
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
            
    i_save_path = './roll_grasp_i_test/'
    img_files_path = []
    for i in range(len(data_for_multi_scene)):
        img_files_path.append(i_save_path+str(int(data_for_multi_scene[i, 15]))+'.png')
    img_files_path = np.array(img_files_path).reshape(-1,1)
    np.save('test_roll_data.npy',np.hstack([data_for_multi_scene[:, :15],img_files_path]))
    
    shutil.rmtree(temp_save_path)
