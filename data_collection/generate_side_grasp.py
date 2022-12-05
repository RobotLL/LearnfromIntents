import random
import shutil
from math import radians, cos, sin, sqrt, atan2
import math
import os
import numpy as np
import pybullet as p
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import concurrent.futures
from tqdm import tqdm
from tqdm import trange
from util import rotate_point, is_point_inside_rectangle, are_angles_close, normalize_angle
from find_negetive_side import get_full_action_table, get_all_good_in_root, sample_class3_from_bad
from util import gripper_limits, objects_limits, generate_side_obj, get_depth_image
from util import normalize_01 as normalize
from PIL import Image 

PLOT_VALID_GRASPS = False


class CommandClass:
    SIDE_GRASP = 1


class SideGrasp:
    TOP_DOWN = 0
    BOTTOM_UP = 1
    LEFT_RIGHT = 2
    RIGHT_LEFT = 3

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
    y_random = random.uniform(-1, 0) * 0.05
    
    
    yaw_random1 = random.uniform(-1, 1) * 30
    yaw_random2 = random.uniform(-1, 0) * 30 - 60
    yaw_random3 = random.uniform(0, 1) * 30 + 60 
    
    yaw_random = math.radians(random.choice([yaw_random1,yaw_random1,yaw_random2,yaw_random3]))
    
    present_object_id = generate_side_obj(obj_position=[
                                          x_random, y_random, 0.03], obj_orientation=p.getQuaternionFromEuler([pitch_random, roll_random, yaw_random]))

    for _ in range(480):
        p.stepSimulation()
    object_pos, object_orn = p.getBasePositionAndOrientation(present_object_id)
    object_orn = p.getEulerFromQuaternion(object_orn)
    
    depth_image = get_depth_image()
    
    p.disconnect()
    return object_pos, object_orn, depth_image

# table depth min=221
# side object min 214
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

def plot_grasp(active_finger_x, active_finger_y, passive_finger_x, passive_finger_y, object_position, object_width, object_length, angle=0, index=0, grasp_side=None, text=None, extra_positions=None, extra_widths=None, extra_lengths=None, extra_angles=None, extra_points=None):
    _, ax = plt.subplots(figsize=(7, 7))
    ax.plot(active_finger_x, active_finger_y, 'ro')
    ax.plot(passive_finger_x, passive_finger_y, 'yo')

    ax.plot(object_position[0], object_position[1], 'bo')

    if text is not None:
        ax.text(0.09, -0.09, text, fontsize=10)

    if grasp_side is not None:
        if grasp_side == SideGrasp.BOTTOM_UP:
            ax.arrow(0.08, -0.08, 0, 0.02)
        elif grasp_side == SideGrasp.TOP_DOWN:
            ax.arrow(0.08, -0.08, 0, -0.02)
        elif grasp_side == SideGrasp.LEFT_RIGHT:
            ax.arrow(0.08, -0.08, 0.02, 0)
        else:
            ax.arrow(0.08, -0.08, -0.02, 0)

    if extra_points is not None:
        for point in extra_points:
            ax.plot(point[0], point[1], 'o', color='#cf954a')

    if object_length > 0:
        # Rectangle
        points = np.array([
            [- object_length / 2, + object_width / 2],
            [+ object_length / 2, + object_width / 2],
            [+ object_length / 2, - object_width / 2],
            [- object_length / 2, - object_width / 2],
        ]).T
        points = rotate_point(points, angle)
        for i in range(4):
            points[:, i] += np.array(object_position[:2])

        rectangle = Polygon(
            points.T,
            edgecolor=None,
            fill=True,
            lw=5)
        ax.add_patch(rectangle)
    else:
        # Circle
        circle = Circle(
            object_position[:2],
            object_width,
            edgecolor=None,
            fill=True,
            lw=5)
        ax.add_patch(circle)

    if extra_positions is not None:
        for extra_position, extra_width, extra_length, extra_angle in zip(extra_positions, extra_widths, extra_lengths, extra_angles):
            points = np.array([
                [- extra_length / 2, + extra_width / 2],
                [+ extra_length / 2, + extra_width / 2],
                [+ extra_length / 2, - extra_width / 2],
                [- extra_length / 2, - extra_width / 2],
            ]).T
            points = rotate_point(points, extra_angle)
            for i in range(4):
                points[:, i] += np.array(extra_position[:2])

            rectangle = Polygon(
                points.T,
                edgecolor='#ffdddd',
                facecolor='#ffdddd',
                fill=True,
                lw=5)
            ax.add_patch(rectangle)

    plt.xlim([-0.13, 0.13])
    plt.ylim([-0.13, 0.13])
    plt.savefig(f'grasps/grasp_{index}.png')
    plt.close()


def generate_side_approaches(object_pos, object_orn, idx):
    one_scene_data = []
    object_width = 0.054
    object_length = 0.0856
    gripper_open_distance = 0.08
    target_distance_from_object_origin = object_width / 2 + 0.02
    gripper_parallel_tolerance = 0.046
    gripper_perpendicular_tolerance = 0.036

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
        [-45],
        [-90, 0, 90],

        # gripper open
        [0]
    ]

    gripper_angle_tolerance = np.pi / 4
    for index, observation in enumerate(itertools.product(*observation_options)):
        object_position = observation[0:3]
        object_orientation_euler = observation[3:6]
        gripper_position = observation[6:9]
        gripper_orientation_euler = observation[9:12]
        open_gripper = observation[12]

        finger_distance = gripper_open_distance * \
            cos(radians(gripper_orientation_euler[1]))

        # Gripper can grasp the object perfectly if the angle distance between object and gripper is 0.
        object_yaw = object_orientation_euler[2]
        gripper_yaw = radians(gripper_orientation_euler[2])

        # Calculate finger positions
        upper_finger_x = gripper_position[0] + 0.5 * \
            finger_distance * cos(gripper_yaw + np.pi / 2)
        upper_finger_y = gripper_position[1] + 0.5 * \
            finger_distance * sin(gripper_yaw + np.pi / 2)
        lower_finger_x = gripper_position[0] + 0.5 * \
            finger_distance * cos(gripper_yaw + np.pi * 3 / 2)
        lower_finger_y = gripper_position[1] + 0.5 * \
            finger_distance * sin(gripper_yaw + np.pi * 3 / 2)

        # Calculate valid regions for a finger to sliding from
        target_gripper_point_upper = [
            object_position[0] + target_distance_from_object_origin *
            cos(object_yaw + np.pi / 2),
            object_position[1] + target_distance_from_object_origin *
            sin(object_yaw + np.pi / 2)
        ]
        target_gripper_point_lower = [
            object_position[0] + target_distance_from_object_origin *
            cos(object_yaw + np.pi * 3 / 2),
            object_position[1] + target_distance_from_object_origin *
            sin(object_yaw + np.pi * 3 / 2)
        ]
        is_lower_inside_upper_rectangle = is_point_inside_rectangle(
            [lower_finger_x, lower_finger_y], target_gripper_point_upper[:2], gripper_perpendicular_tolerance, gripper_parallel_tolerance, object_yaw)
        is_lower_inside_lower_rectangle = is_point_inside_rectangle(
            [lower_finger_x, lower_finger_y], target_gripper_point_lower[:2], gripper_perpendicular_tolerance, gripper_parallel_tolerance, object_yaw)
        is_upper_inside_upper_rectangle = is_point_inside_rectangle(
            [upper_finger_x, upper_finger_y], target_gripper_point_upper[:2], gripper_perpendicular_tolerance, gripper_parallel_tolerance, object_yaw)
        is_upper_inside_lower_rectangle = is_point_inside_rectangle(
            [upper_finger_x, upper_finger_y], target_gripper_point_lower[:2], gripper_perpendicular_tolerance, gripper_parallel_tolerance, object_yaw)
        is_lower_inside_rectangle = is_lower_inside_upper_rectangle or is_lower_inside_lower_rectangle
        is_upper_inside_rectangle = is_upper_inside_upper_rectangle or is_upper_inside_lower_rectangle

        # The gripper is symetrical, handle both cases
        for i in range(2):
            positive_pitch = False
            active_finger_position = None
            passive_finger_position = None

            if i == 0:
                active_finger_position = [upper_finger_x, upper_finger_y]
                passive_finger_position = [lower_finger_x, lower_finger_y]
            else:
                active_finger_position = [lower_finger_x, lower_finger_y]
                passive_finger_position = [upper_finger_x, upper_finger_y]
                positive_pitch = True

            target_gripper_yaw = object_yaw
            if is_lower_inside_upper_rectangle or is_upper_inside_lower_rectangle:
                target_gripper_yaw += np.pi
            target_gripper_yaw = normalize_angle(target_gripper_yaw)

            # Determine the slide direction
            grasp_side = None
            is_object_on_correct_side = False
            move_direction = atan2(passive_finger_position[1] - active_finger_position[1],
                                   passive_finger_position[0] - active_finger_position[0])
            if radians(-135) <= move_direction < radians(-45):
                grasp_side = SideGrasp.TOP_DOWN
                if object_position[1] <= 0:
                    is_object_on_correct_side = True
            elif radians(-45) <= move_direction < radians(45):
                grasp_side = SideGrasp.LEFT_RIGHT
                if object_position[0] >= 0:
                    is_object_on_correct_side = True
            elif radians(45) <= move_direction < radians(135):
                grasp_side = SideGrasp.BOTTOM_UP
                if object_position[1] >= 0:
                    is_object_on_correct_side = True
            else:
                grasp_side = SideGrasp.RIGHT_LEFT
                if object_position[0] <= 0:
                    is_object_on_correct_side = True

            # Check whether the grasp is valid
            valid = are_angles_close(target_gripper_yaw, gripper_yaw, gripper_angle_tolerance) and \
                ((i == 1 and is_lower_inside_rectangle) or (i == 0 and is_upper_inside_rectangle)) and \
                is_object_on_correct_side and \
                (gripper_orientation_euler[1] > 0) == positive_pitch
            if valid:
                if PLOT_VALID_GRASPS:
                    plot_grasp(
                        active_finger_position[0],
                        active_finger_position[1],
                        passive_finger_position[0],
                        passive_finger_position[1],
                        object_position,
                        object_width,
                        object_length,
                        angle=object_yaw,
                        index=index,
                        grasp_side=grasp_side,
                        text=f'{target_gripper_yaw:.2} / {object_yaw:.2} {gripper_yaw:.2}',
                        extra_points=[
                            gripper_position[:2]
                        ],
                        extra_positions=[
                            target_gripper_point_upper,
                            target_gripper_point_lower,
                        ],
                        extra_widths=[gripper_perpendicular_tolerance,
                                      gripper_perpendicular_tolerance],
                        extra_lengths=[gripper_parallel_tolerance,
                                       gripper_parallel_tolerance],
                        extra_angles=[object_yaw, object_yaw],
                    )

                one_scene_data.append([
                    *object_position,
                    *object_orientation_euler,
                    *gripper_position,
                    *gripper_orientation_euler,
                    open_gripper,
                    CommandClass.SIDE_GRASP,
                    idx,
                    grasp_side,
                ])

    return np.array(one_scene_data)


def collect_data(save_data_name, c_id, for_train = True):

    if PLOT_VALID_GRASPS:
        if os.path.exists('grasps'):
            shutil.rmtree('grasps')
        os.mkdir('grasps')
    
    if for_train == True:
        i_save_path = './side_grasp_i_train/'
        if os.path.exists(i_save_path):
            pass
        else:
            os.mkdir('side_grasp_i_train')
        
    else:
        i_save_path = './side_grasp_i_test/'
        if os.path.exists(i_save_path):
            pass
        else:
            os.mkdir('side_grasp_i_test')
        
    
    i = c_id
    # print('collect_number: ', i)
    object_pos, object_orn, depth_i = generate_random_object_pose()
    root_c1 = generate_side_approaches(object_pos, object_orn, idx = i)
    # print(len(root_c1))
    while len(root_c1) == 0:
        object_pos, object_orn, depth_i = generate_random_object_pose()
        root_c1 = generate_side_approaches(object_pos, object_orn, idx = i)
    
    
    Image.fromarray(depth_i).convert('L').save(i_save_path+str(i)+'.png')
    
    full_table = get_full_action_table(object_pos, object_orn)
    good_table = get_all_good_in_root(root_c1, object_pos, object_orn)
    #todo: add idx to bad table
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

#%%
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
    
    i_save_path = './side_grasp_i_train/'
    img_files_path = []
    for i in range(len(data_for_multi_scene)):
        img_files_path.append(i_save_path+str(int(data_for_multi_scene[i, 15]))+'.png')
    img_files_path = np.array(img_files_path).reshape(-1,1)
    
    np.save('train_side_data.npy',np.hstack([data_for_multi_scene[:, :15],img_files_path]))
    
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
            
    i_save_path = './side_grasp_i_test/'
    img_files_path = []
    for i in range(len(data_for_multi_scene)):
        img_files_path.append(i_save_path+str(int(data_for_multi_scene[i, 15]))+'.png')
    img_files_path = np.array(img_files_path).reshape(-1,1)
    
    np.save('test_side_data.npy',np.hstack([data_for_multi_scene[:, :15],img_files_path]))
    
    shutil.rmtree(temp_save_path)

# def main():
#     # collect train data, collect_number is scene number not grasp number
#     collect_data('train_data.csv', collect_number=2000)
#     # collect test data
#     collect_data('test_data.csv', collect_number=200)
# if __name__ == '__main__':
#     main()
