import numpy as np
import pandas as pd
import itertools
import random


def get_full_action_table(object_pos, object_orn):
    observation_options = [
        # object position
        [object_pos[0]], [object_pos[1]], [object_pos[2]],
        # object orientation
        [object_orn[0]], [object_orn[1]], [object_orn[2]],
        # gripper start position
        [-0.1, -0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.1],
        [-0.1, -0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.1],
        [0, 0.05, 0.1],
        # gripper start orientation
        [-45, 0, 45],
        [-45, 0, 45],
        [-90, -60, -30,   0,  30,  60, 90],
        # gripper open
        [0, 1],
    ]
    full_table = []
    for observation in itertools.product(*observation_options):
        full_table.append(observation)
    full_table = np.around(full_table, decimals=3)
    return full_table


def get_all_good_in_root(root_data, object_pos, object_orn):
    # 15 col
    # object_x,object_y,object_z,object_roll,object_pitch,object_yaw,
    # gripper_x,gripper_y,gripper_z,gripper_roll,gripper_pitch,gripper_yaw,
    # open_gripper
    # command_class
    # push_direction
    root_c1 = np.around(np.array(root_data), decimals=3)[:, :15]
    push_direction = np.around(np.array(root_data), decimals=3)[:, 15]
    # calculate class c2
    root_c2 = root_c1.copy()
    if push_direction[0] == 0:
        root_c2[:, 7] = -0.1
    elif push_direction[0] == 1:
        root_c2[:, 7] = 0.1
    elif push_direction[0] == 2:
        root_c2[:, 6] = 0.1
    elif push_direction[0] == 3:
        root_c2[:, 6] = -0.1
    else:
        print('wrong')
        exit()
    # classid=2
    root_c2[:, 13] = 2

    # calculate class c3
    root_c3 = root_c2.copy()
    # negetive pitch
    root_c3[:, 10] = root_c3[:, 10]*-1
    # gripper close
    root_c3[:, 12] = 1
    # classid = 3
    root_c3[:, 13] = 3

    # calculate class c0
    root_c0 = np.array([object_pos[0], object_pos[1], object_pos[2], object_orn[0], object_orn[1], object_orn[2], 0, 0, 0.5,
                       0, 0, 0, 0, 0, root_c1[0][14]]).reshape(1, 15)
    
    root_c0 = np.around(root_c0, decimals=3)

    good_table = np.vstack([root_c3, root_c2, root_c1, root_c0])
    return good_table


def sample_class3_from_bad(full_table, good_table):
    good_without_class = good_table[:, 0:13]
    is_in = (full_table[:, None] == good_without_class).all(-1).any(-1)
    bad_index_list = []
    for i in range(len(is_in)):
        if np.all(is_in[i]) == False:
            bad_index_list.append(i)
    # 1% bad grasp
    bad_table = full_table[random.sample(
        bad_index_list, int(len(bad_index_list)/100))]

    # set bad_class_id
    bad_class = np.ones(len(bad_table)).reshape(len(bad_table), 1)*4
    bad_table = np.hstack([bad_table, bad_class])
    #
    bad_idx = np.ones(len(bad_table)).reshape(len(bad_table), 1)*good_table[0, 14]
    bad_table = np.hstack([bad_table, bad_idx])
    return bad_table
