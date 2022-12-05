import numpy as np
import gym
import pybullet as p
import random
import math
import time
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import torch
from util import normalize_01 as normalize
from util import (
    normalize_01,
    objects_limits,
    gripper_orn_to_world,
    gripper_limits,
    discretize,
)
from util import generate_side_obj, generate_roll_obj, generate_td_obj

ACTION_SIZE = 7  # x y z roll pitch yaw if_open
VECTOR_SIZE = 8
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120
CAMERA_FAR = 1
CAMERA_NEAR = 0.2
HFOV_VFOV = 320 / 240


class Sim(gym.Env):
    def __init__(
        self,
        gui=False,
        discrete=True,
        number_of_objects=1,
        reset_interval=1,
        classification_model=None,
    ):

        # surface height
        self.table_urdf = "./model/env/table.urdf"
        self.table_position = [0, 0, -0.05 / 2]

        # robot and objects
        self.robot_position = [0, 0, 0.5]  # make sure not hide camera
        self.reset_every = reset_interval

        self.gripper_urdf = "./model/robot/topdown_gripper.urdf"

        # self.object_number = number_of_objects
        self.max_object_height = 0.2
        # var
        self.joints = {}
        self.objects = []
        self.gripper_orn_obs = [0, 0, 0]
        self.gripper_distance = 0
        self.if_open = 0
        # rl relative
        self.steps = 0
        self.episode = 0
        self.penetration = False

        # camera
        self.depth_image = None
        self.use_gui = gui
        self.discrete = discrete

        if self.use_gui:
            p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.3,
                cameraYaw=50.8,
                cameraPitch=-44.2,
                cameraTargetPosition=[-0.56, 0.47, -0.52],
            )
        else:
            p.connect(p.DIRECT)
        self.observation_space = self.get_observation_space()
        if self.discrete is True:
            # order xyz rpy if_open
            self.action_space = gym.spaces.MultiDiscrete(
                [11, 11, 3, 3, 3, 7, 2]
            )  # mul change
        else:
            self.action_space = self.get_action_space()
        self.__classification_model = classification_model

    def set_environment_properties(self, target_number_of_objects, workspace_length):
        # self.object_number = target_number_of_objects
        pass

    def get_observation_space(self):
        return gym.spaces.Dict(
            {
                "vector": gym.spaces.Box(
                    low=-1, high=1, shape=(VECTOR_SIZE,), dtype=np.float32
                ),
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(1, 60, 60), dtype=np.uint8
                ),
            }
        )

    def get_action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(ACTION_SIZE,), dtype=np.float32)

    def lift_gripper(self, distance=0.15, initial_orientation=None):
        state = p.getLinkState(self.robot_id, self.joints["joint_yaw"].id)
        robot_pos = state[0]
        robot_orn = state[1] if initial_orientation is None else initial_orientation
        jointPose = p.calculateInverseKinematics(
            self.robot_id,
            self.control_link_index,
            [robot_pos[0], robot_pos[1], robot_pos[2] + distance],
            robot_orn,
        )
        for jointName in self.joints:
            if jointName in self.position_control_joint_name:
                joint = self.joints[jointName]
                p.setJointMotorControl2(
                    self.robot_id,
                    joint.id,
                    p.POSITION_CONTROL,
                    targetPosition=jointPose[joint.id],
                    force=joint.maxForce,
                    maxVelocity=0.15,
                )

    def get_center_pose(self):
        state = p.getLinkState(self.robot_id, self.joints["joint_yaw"].id)
        return state[0], state[1]

    def _delay(self, duration):
        for _ in range(duration):
            if self.use_gui is True:
                time.sleep(0.005)
            p.stepSimulation()

    def spawnobjects(self):
        self.delete_fallenobjects()

        pitch_random = 0
        roll_random = 0
        x_random = random.uniform(-1, 1) * 0.025
        y_random = random.uniform(-1, 1) * 0.025
        yaw_random = random.uniform(-1, 1) * 1.57

        if self.object_type == 0:
            self.drop_height = 0.03
            self.reward_height = 0.03
            uid = generate_side_obj(
                [x_random, y_random, self.drop_height],
                p.getQuaternionFromEuler(
                    [pitch_random, roll_random, yaw_random]),
            )
            self.objects.append(uid)
            self._delay(50)
        elif self.object_type == 1:
            self.drop_height = 0.05
            self.reward_height = 0.05
            uid = generate_td_obj(
                [x_random, y_random, self.drop_height],
                p.getQuaternionFromEuler(
                    [pitch_random, roll_random, yaw_random]),
            )
            self.objects.append(uid)
            self._delay(50)
        elif self.object_type == 2:
            self.drop_height = 0.055
            self.reward_height = 0.09
            uid = generate_roll_obj(
                [x_random, y_random, self.drop_height],
                p.getQuaternionFromEuler(
                    [pitch_random, roll_random, yaw_random]),
            )
            self.objects.append(uid)
            self._delay(50)

        elif self.object_type == 3:
            number_of_objects = random.randint(2, 4)
            self.drop_height = 0.055
            self.reward_height = 0.07
            position_lists = [0, 1, 2, 3]
            positions = random.sample(position_lists, number_of_objects)
            pitch_random = 0
            roll_random = 0
            for pos_id in positions:
                if pos_id == 0:
                    x_random = 0.025 + random.random()*0.025
                    y_random = 0.025 + random.random()*0.025
                    yaw_random = random.uniform(-1, 1) * 1.57

                    obj_type = random.randint(1, 2)
                    if obj_type == 1:
                        uid = generate_td_obj(
                            [x_random, y_random, self.drop_height],
                            p.getQuaternionFromEuler(
                                [pitch_random, roll_random, yaw_random]),
                        )
                    else:
                        uid = generate_roll_obj(
                            [x_random, y_random, self.drop_height],
                            p.getQuaternionFromEuler(
                                [pitch_random, roll_random, yaw_random]),
                        )
                    self.objects.append(uid)

                elif pos_id == 1:
                    x_random = 0.025 + random.random()*0.025
                    y_random = -0.025 - random.random()*0.025
                    yaw_random = random.uniform(-1, 1) * 1.57
                    obj_type = random.randint(1, 2)
                    if obj_type == 1:
                        uid = generate_td_obj(
                            [x_random, y_random, self.drop_height],
                            p.getQuaternionFromEuler(
                                [pitch_random, roll_random, yaw_random]),
                        )
                    else:
                        uid = generate_roll_obj(
                            [x_random, y_random, self.drop_height],
                            p.getQuaternionFromEuler(
                                [pitch_random, roll_random, yaw_random]),
                        )
                    self.objects.append(uid)

                elif pos_id == 2:
                    x_random = -0.025 - random.random()*0.025
                    y_random = 0.025 + random.random()*0.025
                    yaw_random = random.uniform(-1, 1) * 1.57
                    obj_type = random.randint(1, 2)
                    if obj_type == 1:
                        uid = generate_td_obj(
                            [x_random, y_random, self.drop_height],
                            p.getQuaternionFromEuler(
                                [pitch_random, roll_random, yaw_random]),
                        )
                    else:
                        uid = generate_roll_obj(
                            [x_random, y_random, self.drop_height],
                            p.getQuaternionFromEuler(
                                [pitch_random, roll_random, yaw_random]),
                        )
                    self.objects.append(uid)

                elif pos_id == 3:
                    x_random = -0.025 - random.random()*0.025
                    y_random = -0.025 - random.random()*0.025
                    yaw_random = random.uniform(-1, 1) * 1.57
                    obj_type = random.randint(1, 2)
                    if obj_type == 1:
                        uid = generate_td_obj(
                            [x_random, y_random, self.drop_height],
                            p.getQuaternionFromEuler(
                                [pitch_random, roll_random, yaw_random]),
                        )
                    else:
                        uid = generate_roll_obj(
                            [x_random, y_random, self.drop_height],
                            p.getQuaternionFromEuler(
                                [pitch_random, roll_random, yaw_random]),
                        )
                    self.objects.append(uid)
            self._delay(50)

    def delete_fallenobjects(self):
        # Delete objects under the table
        for object_id in self.objects:
            position, _ = p.getBasePositionAndOrientation(object_id)
            if position[2] < -0.01 or position[2] > self.max_object_height:
                self.objects.remove(object_id)
                p.removeBody(object_id)

    def control_gripper(
        self, world_position, world_orientation, velocity=None, j_force=None
    ):
        jointPose = p.calculateInverseKinematics(
            self.robot_id, self.control_link_index, world_position, world_orientation
        )
        for jointName in self.joints:
            if jointName in self.position_control_joint_name:
                joint = self.joints[jointName]
                joint_velocity = joint.maxVelocity if velocity is None else velocity
                joint_force = joint.maxForce if j_force is None else j_force
                if jointName in ["joint_yaw", "joint_pitch", "joint_roll"]:
                    joint_velocity = joint.maxVelocity
                    joint_force = joint.maxForce
                p.setJointMotorControl2(
                    self.robot_id,
                    joint.id,
                    p.POSITION_CONTROL,
                    targetPosition=jointPose[joint.id],
                    force=joint_force,
                    maxVelocity=joint_velocity,
                )

    def close_gripper(self):
        # 1mm distace
        p.setJointMotorControl2(
            self.robot_id,
            self.joints[self.right_finger_joint_name].id,
            p.POSITION_CONTROL,
            targetPosition=-0.001,
            force=70,
            maxVelocity=0.025,
        )
        p.setJointMotorControl2(
            self.robot_id,
            self.joints[self.left_finger_joint_name].id,
            p.POSITION_CONTROL,
            targetPosition=0.001,
            force=70,
            maxVelocity=0.025,
        )

    def open_gripper(self, ditance_between_fingers):
        # ditance_between_fingers unit is cm
        ditance_between_fingers = ditance_between_fingers / 100 / 2
        p.setJointMotorControl2(
            self.robot_id,
            self.joints[self.right_finger_joint_name].id,
            p.POSITION_CONTROL,
            targetPosition=-ditance_between_fingers,
            force=500,
            maxVelocity=0.1,
        )
        p.setJointMotorControl2(
            self.robot_id,
            self.joints[self.left_finger_joint_name].id,
            p.POSITION_CONTROL,
            targetPosition=ditance_between_fingers,
            force=500,
            maxVelocity=0.1,
        )

    def render_camera(self):
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_position,
            cameraTargetPosition=self.camera_target,
            cameraUpVector=[0, 1, 0],
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=HFOV_VFOV,
            nearVal=CAMERA_NEAR,
            farVal=CAMERA_FAR,
        )
        _, _, _, depth_image, _ = p.getCameraImage(
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        # Depth image
        depth_image = (
            CAMERA_FAR
            * CAMERA_NEAR
            / (CAMERA_FAR - (CAMERA_FAR - CAMERA_NEAR) * depth_image)
        )
        # Add noise
        # noise = np.random.normal(0, 1, [IMAGE_HEIGHT, IMAGE_WIDTH])
        # noise = noise/(np.max(noise) - np.min(noise))/100/2  # 5mm noise
        # depth_image = depth_image + noise

        depth_image = np.array(depth_image).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
        min_in_d = 0.3
        max_in_d = 0.41

        process_depth = depth_image.copy()
        process_depth = (process_depth - min_in_d) / \
            (max_in_d - min_in_d) * 255
        process_depth[np.where(depth_image > 0.5)] = 255

        cp_w = 80
        cp_h = 60
        process_depth = process_depth[cp_h -
                                      30: cp_h + 30, cp_w - 30: cp_w + 30]
        process_depth = np.array(process_depth).astype(np.uint8)
        process_depth = process_depth[np.newaxis, :, :]
        process_depth = np.array(process_depth).astype(np.uint8)
        self.depth_image = process_depth

    def get_observation(self, x, y, z, roll, pitch, yaw, current_if_open):

        current_x = normalize(x, gripper_limits["joint_x"])
        current_y = normalize(y, gripper_limits["joint_y"])
        current_z = normalize(z, gripper_limits["joint_z"])

        current_roll = normalize(roll, gripper_limits["joint_roll"])
        current_pitch = normalize(pitch, gripper_limits["joint_pitch"])
        current_yaw = normalize(yaw, gripper_limits["joint_yaw"])
        current_if_open = normalize(
            current_if_open, gripper_limits["joint_open"])

        current_step = normalize(self.steps, gripper_limits["steps"])

        gripper_pose = [
            current_x,
            current_y,
            current_z,
            current_roll,
            current_pitch,
            current_yaw,
            current_if_open,
            current_step,
        ]
        return {
            "vector": np.array(gripper_pose, dtype=np.float32),
            "image": self.depth_image,
        }

    def get_real_vector(self, x, y, z, roll, pitch, yaw, current_if_open, steps):
        # for real test
        current_x = normalize(x, gripper_limits["joint_x"])
        current_y = normalize(y, gripper_limits["joint_y"])
        current_z = normalize(z, gripper_limits["joint_z"])

        current_roll = normalize(roll, gripper_limits["joint_roll"])
        current_pitch = normalize(pitch, gripper_limits["joint_pitch"])
        current_yaw = normalize(yaw, gripper_limits["joint_yaw"])
        current_if_open = normalize(
            current_if_open, gripper_limits["joint_open"])

        current_step = normalize(steps, gripper_limits["steps"])

        gripper_pose = [
            current_x,
            current_y,
            current_z,
            current_roll,
            current_pitch,
            current_yaw,
            current_if_open,
            current_step,
        ]
        return np.array(gripper_pose, dtype=np.float32)

    def build_environment(self):

        p.resetSimulation()

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(numSolverIterations=50)

        # Define world
        p.setGravity(0, 0, -9.8)
        # p.loadURDF('plane.urdf', [0, 0, 0], useFixedBase=True)

        # Define box
        self.table_id = p.loadURDF(
            self.table_urdf,
            self.table_position,
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )

        # Define the robot
        self.robot_id = p.loadURDF(
            self.gripper_urdf,
            self.robot_position,
            [0, 0, 0, 1],
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
        )

        # Store joint information
        jointTypeList = ["REVOLUTE", "PRISMATIC",
                         "SPHERICAL", "PLANAR", "FIXED"]
        self.number_of_joints = p.getNumJoints(self.robot_id)
        jointInfo = namedtuple(
            "jointInfo",
            [
                "id",
                "name",
                "type",
                "lowerLimit",
                "upperLimit",
                "maxForce",
                "maxVelocity",
            ],
        )

        self.joints = AttrDict()
        self.control_link_index = 0
        # get jointInfo and index of dummy_center_indicator_link
        for i in range(self.number_of_joints):
            info = p.getJointInfo(self.robot_id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = jointTypeList[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            singleInfo = jointInfo(
                jointID,
                jointName,
                jointType,
                jointLowerLimit,
                jointUpperLimit,
                jointMaxForce,
                jointMaxVelocity,
            )
            self.joints[singleInfo.name] = singleInfo
            # register index of dummy center link
            if jointName == "joint_yaw":
                self.control_link_index = i
        self.position_control_joint_name = [
            "joint_x",
            "joint_y",
            "joint_z",
            "joint_roll",
            "joint_pitch",
            "joint_yaw",
        ]
        self.left_finger_joint_name = "joint_left_finger"
        self.right_finger_joint_name = "joint_right_finger"

    def reset(self):
        self.steps = 0
        self.episode += 1
        self.target_obj = None

        self.penetration = False
        # Sometimes reset the environment
        if self.episode % self.reset_every == 0 or self.episode <= 1:
            # bin: 3
            self.object_type = random.choice([0, 1, 2, 3])
            #
            self.objects.clear()
            self.gripper_orn_obs = [0, 0, 0]
            self.gripper_distance = 0
            self.if_open = 0

            self.build_environment()
            while True:
                self.delete_fallenobjects()
                if 0 < len(self.objects):
                    break
                else:
                    self.spawnobjects()
        else:
            p.removeBody(self.robot_id)
            self.robot_id = p.loadURDF(
                self.gripper_urdf,
                self.robot_position,
                [0, 0, 0, 1],
                flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
            )
            while True:
                self.delete_fallenobjects()
                if 0 < len(self.objects):
                    break
                else:
                    self.spawnobjects()
        self.camera_position = [0, 0, 0.4]
        self.camera_target = [0, 0, 0]
        self.camera_fov = 55.371784673413806
        self.camera_position[0] = (
            self.camera_position[0] + random.uniform(-1, 1) * 0.005
        )
        self.camera_position[1] = (
            self.camera_position[1] + random.uniform(-1, 1) * 0.005
        )
        self.camera_target[0] = self.camera_target[0] + \
            random.uniform(-1, 1) * 0.005
        self.camera_target[1] = self.camera_target[1] + \
            random.uniform(-1, 1) * 0.005
        # self.camera_fov = 55.371784673413806 + random.uniform(-0.1, 0.1)

        self.render_camera()
        (
            self.object_position,
            self.object_orientation_quaternion,
        ) = p.getBasePositionAndOrientation(self.objects[0])
        self.object_orientation_euler = p.getEulerFromQuaternion(
            self.object_orientation_quaternion
        )
        return self.get_observation(0, 0, 0.1, 0, 0, 0, 0)

    def get_probs(self, o_xyz, o_rpy, g_xyz, g_rpy, if_open, dis):

        object_position = np.round(o_xyz, decimals=3)
        object_orientation = np.round(o_rpy, decimals=3)
        gripper_position = g_xyz
        gripper_orientation = g_rpy

        object_position[0] = normalize_01(
            object_position[0], objects_limits["o_x"])
        object_position[1] = normalize_01(
            object_position[1], objects_limits["o_y"])
        # todo: only for slim
        # object_position[2] = 0.002
        object_position[2] = normalize_01(
            object_position[2], objects_limits["o_z"])

        object_orientation[0] = normalize_01(
            object_orientation[0], objects_limits["o_roll"]
        )
        object_orientation[1] = normalize_01(
            object_orientation[1], objects_limits["o_pitch"]
        )
        object_orientation[2] = normalize_01(
            object_orientation[2], objects_limits["o_yaw"]
        )

        gripper_position[0] = normalize_01(
            gripper_position[0], gripper_limits["joint_x"]
        )
        gripper_position[1] = normalize_01(
            gripper_position[1], gripper_limits["joint_y"]
        )
        gripper_position[2] = normalize_01(
            gripper_position[2], gripper_limits["joint_z"]
        )
        gripper_orientation[0] = normalize_01(
            gripper_orientation[0], gripper_limits["joint_roll"]
        )
        gripper_orientation[1] = normalize_01(
            gripper_orientation[1], gripper_limits["joint_pitch"]
        )
        gripper_orientation[2] = normalize_01(
            gripper_orientation[2], gripper_limits["joint_yaw"]
        )
        if_open = normalize_01(if_open, gripper_limits["joint_open"])
        dis = normalize_01(dis, gripper_limits["joint_distance"])

        vector = [
            *object_position,
            *object_orientation,
            *gripper_position,
            *gripper_orientation,
            *[if_open],
            *[dis],
        ]
        vector = np.array(vector)[np.newaxis, :]
        classifier_input = torch.Tensor(vector)

        image = np.array(self.depth_image)[np.newaxis, :, :, :]
        image = torch.Tensor(image)

        classifier_raw_output = (
            self.__classification_model.forward(classifier_input, image)
            .detach()
            .numpy()
        )
        classifier_raw_output = np.squeeze(classifier_raw_output)
        max_raw = np.max(classifier_raw_output)
        probs = np.exp(classifier_raw_output - max_raw) / sum(
            np.exp(classifier_raw_output - max_raw)
        )
        # classifier_exp_output = np.exp(classifier_raw_output)
        # probs = classifier_exp_output / np.sum(classifier_exp_output)
        return probs

    def get_dis(self, ox, oy, gx, gy):
        distance = np.round(
            np.sqrt(np.square(ox - gx) + np.square(oy - gy)), decimals=3
        )
        return distance

    def step(self, action):

        done = False
        self.steps += 1
        reward = 0
        observation = None
        if self.discrete is True:
            new_x = (action[0] * 2 - 10) * 0.01
            new_y = (action[1] * 2 - 10) * 0.01
            new_z = action[2] * 0.05

            roll = action[3] * 45 - 45
            pitch = action[4] * 45 - 45
            yaw = action[5] * 30 - 90
            self.if_open = action[6]
        else:
            new_x = np.round(
                discretize(
                    action[0] * 0.1,
                    [-0.1, -0.08, -0.06, -0.04, -0.02,
                        0, 0.02, 0.04, 0.06, 0.08, 0.1],
                ),
                decimals=2,
            )
            new_y = np.round(
                discretize(
                    action[1] * 0.1,
                    [-0.1, -0.08, -0.06, -0.04, -0.02,
                        0, 0.02, 0.04, 0.06, 0.08, 0.1],
                ),
                decimals=2,
            )
            new_z = np.round(
                discretize((action[2] + 1) / 2 * 0.1, [0, 0.05, 0.1]), decimals=2
            )

            roll = np.round(discretize(
                action[3] * 45, [-45, 0, 45]), decimals=0)
            pitch = np.round(discretize(
                action[4] * 45, [-45, 0, 45]), decimals=0)
            yaw = np.round(
                discretize(action[5] * 90, [-90, -60, -30, 0, 30, 60, 90]), decimals=0
            )

            self.if_open = np.round(discretize(
                (action[6] + 1) / 2, [0, 1]), decimals=0)
        if pitch == 45 or pitch == -45:
            tolerrance = 0.032
        else:
            tolerrance = 0
        self.gripper_distance = 8
        self.gripper_orn_obs = gripper_orn_to_world(
            math.radians(pitch), math.radians(roll), math.radians(yaw)
        )

        observation = self.get_observation(
            new_x, new_y, new_z, roll, pitch, yaw, self.if_open
        )

        if self.object_type == 0:

            if self.__classification_model is not None:
                dis = self.get_dis(
                    self.object_position[0], self.object_position[1], new_x, new_y
                )
                probs = self.get_probs(
                    o_xyz=[
                        self.object_position[0],
                        self.object_position[1],
                        self.object_position[2],
                    ],
                    o_rpy=[
                        self.object_orientation_euler[0],
                        self.object_orientation_euler[1],
                        self.object_orientation_euler[2],
                    ],
                    g_xyz=[new_x, new_y, new_z],
                    g_rpy=[roll, pitch, yaw],
                    if_open=self.if_open,
                    dis=dis,
                )

                reward = min(probs[self.steps], 0.99)
                # reward = min(probs[self.steps], 0.99)

                if self.steps == 1:
                    # reward = min(probs[self.steps], 0.99)
                    # self.r1 = min(probs[self.steps], 0.99)

                    p.removeBody(self.robot_id)
                    self.robot_id = p.loadURDF(
                        self.gripper_urdf,
                        [new_x, new_y, new_z + tolerrance],
                        self.gripper_orn_obs,
                        flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                    )
                    self.open_gripper(self.gripper_distance)
                    self._delay(240)
                    if len(p.getContactPoints(self.robot_id)) != 0:
                        self.penetration = True
                        reward = 0
                    # self.y = yaw
                    # if self.check_if(self.object_orientation_euler, yaw) is False:
                    #     reward = 0

                elif self.steps == 2:
                    # self.r2 = min(probs[self.steps], 0.99)
                    # reward = min(probs[self.steps], 0.99)
                    # if self.check_if(self.object_orientation_euler, yaw) is False:
                    #     reward = 0

                    self.control_gripper(
                        [new_x, new_y, new_z + tolerrance],
                        self.gripper_orn_obs,
                        velocity=0.05,
                        j_force=100,
                    )
                    self.open_gripper(self.gripper_distance)
                    self._delay(480)

                    if len(p.getContactPoints(self.table_id, self.robot_id)) != 0:
                        self.penetration = True
                        reward = 0
                    # if self.y != yaw:
                    #     self.penetration = True
                    #     reward = 0

                    # if self.check_if(self.object_orientation_euler, yaw) is False:
                    #     reward = 0

                elif self.steps == 3:
                    # self.r3 = min(probs[self.steps], 0.99)
                    # reward = min(probs[self.steps], 0.99)
                    # if self.check_if(self.object_orientation_euler, yaw) is False:
                    #     reward = 0

                    if pitch == -45:
                        pitch = -60
                    elif pitch == 45:
                        pitch = 60
                    self.gripper_orn_obs = gripper_orn_to_world(
                        math.radians(pitch), math.radians(
                            roll), math.radians(yaw)
                    )

                    self.control_gripper(
                        [new_x, new_y, new_z],
                        self.gripper_orn_obs,
                        velocity=0.05,
                        j_force=100,
                    )
                    self.open_gripper(self.gripper_distance)
                    self._delay(240)

                    if len(p.getContactPoints(self.table_id, self.robot_id)) != 0 or pitch == 0:
                        self.penetration = True
                        reward = 0
                    # if self.y != yaw:
                    #     self.penetration = True
                    #     reward = 0
                    if self.penetration == False and self.check_if(self.object_orientation_euler, yaw) is True:
                        for _ in range(4):
                            self.close_gripper()
                            self._delay(120)
                        self.close_gripper()
                        self.lift_gripper(0.2)
                        self._delay(240)
                        reward += self.get_reward()
                    done = True
        elif self.object_type == 1 or 2:
            if self.__classification_model is not None:
                dis = self.get_dis(
                    self.object_position[0], self.object_position[1], new_x, new_y
                )
                probs = self.get_probs(
                    o_xyz=[
                        self.object_position[0],
                        self.object_position[1],
                        self.object_position[2],
                    ],
                    o_rpy=[
                        self.object_orientation_euler[0],
                        self.object_orientation_euler[1],
                        self.object_orientation_euler[2],
                    ],
                    g_xyz=[new_x, new_y, new_z],
                    g_rpy=[roll, pitch, yaw],
                    if_open=self.if_open,
                    dis=dis,
                )
                reward = min(probs[self.steps], 0.99)

                # Apply action
            if self.steps == 1:
                p.removeBody(self.robot_id)
                self.robot_id = p.loadURDF(
                    self.gripper_urdf,
                    [new_x, new_y, new_z + tolerrance],
                    self.gripper_orn_obs,
                    flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                )
                self.open_gripper(self.gripper_distance)
                self._delay(240)

                if len(p.getContactPoints(self.robot_id)) != 0:
                    self.penetration = True
                    reward = 0
                    done = True
            elif self.steps == 2:

                self.control_gripper(
                    [new_x, new_y, new_z + tolerrance],
                    self.gripper_orn_obs,
                    velocity=0.05,
                    j_force=100,
                )
                self.open_gripper(self.gripper_distance)
                self._delay(480)

                if len(p.getContactPoints(self.table_id, self.robot_id)) != 0:
                    self.penetration = True
                    reward = 0
                    done = True
            elif self.steps == 3:

                if self.object_type == 0:
                    if pitch == 0 or pitch == -45 or self.if_open == 0:
                        done = True
                        reward = 0
                    else:
                        if pitch == -45:
                            pitch = -60
                        elif pitch == 45:
                            pitch = 60
                        self.gripper_orn_obs = gripper_orn_to_world(
                            math.radians(pitch), math.radians(
                                roll), math.radians(yaw)
                        )

                        self.control_gripper(
                            [new_x, new_y, new_z],
                            self.gripper_orn_obs,
                            velocity=0.05,
                            j_force=100,
                        )
                        self.open_gripper(self.gripper_distance)
                        self._delay(240)

                        if self.if_open == 1:
                            for _ in range(4):
                                self.close_gripper()
                                self._delay(120)
                            self.close_gripper()
                            self.lift_gripper(0.2)
                            self._delay(240)
                            reward += self.get_reward()
                        done = True
                else:
                    if pitch == -45:
                        pitch = -60
                    elif pitch == 45:
                        pitch = 60
                    self.gripper_orn_obs = gripper_orn_to_world(
                        math.radians(pitch), math.radians(
                            roll), math.radians(yaw)
                    )

                    self.control_gripper(
                        [new_x, new_y, new_z],
                        self.gripper_orn_obs,
                        velocity=0.05,
                        j_force=100,
                    )
                    self.open_gripper(self.gripper_distance)
                    self._delay(240)

                    if self.if_open == 1:
                        for _ in range(4):
                            self.close_gripper()
                            self._delay(120)
                        self.close_gripper()
                        self.lift_gripper(0.2)
                        self._delay(240)
                        reward += self.get_reward()
                    done = True
        elif self.object_type == 3:
            # Apply action
            if self.steps == 1:
                p.removeBody(self.robot_id)
                self.robot_id = p.loadURDF(
                    self.gripper_urdf,
                    [new_x, new_y, new_z + tolerrance],
                    self.gripper_orn_obs,
                    flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                )
                self.open_gripper(self.gripper_distance)
                self._delay(240)

                if len(p.getContactPoints(self.robot_id)) != 0:
                    self.penetration = True
                    reward = 0
                    done = True
            elif self.steps == 2:

                self.control_gripper(
                    [new_x, new_y, new_z + tolerrance],
                    self.gripper_orn_obs,
                    velocity=0.05,
                    j_force=100,
                )
                self.open_gripper(self.gripper_distance)
                self._delay(480)

                if len(p.getContactPoints(self.table_id, self.robot_id)) != 0:
                    self.penetration = True
                    reward = 0
                    done = True
            elif self.steps == 3:
                if pitch == -45:
                    pitch = -60
                elif pitch == 45:
                    pitch = 60
                self.gripper_orn_obs = gripper_orn_to_world(
                    math.radians(pitch), math.radians(
                        roll), math.radians(yaw)
                )

                self.control_gripper(
                    [new_x, new_y, new_z],
                    self.gripper_orn_obs,
                    velocity=0.05,
                    j_force=100,
                )
                self.open_gripper(self.gripper_distance)
                self._delay(240)

                if self.if_open == 1:
                    for _ in range(4):
                        self.close_gripper()
                        self._delay(120)
                    self.close_gripper()
                    self.lift_gripper(0.2)
                    self._delay(240)
                    reward += self.get_reward()
                done = True
        return observation, reward, done, {}

    def get_reward(self):
        reward = 0
        total_picked = 0
        for object_id in self.objects:
            object_position, _ = p.getBasePositionAndOrientation(object_id)
            if object_position[2] > self.reward_height:
                self.objects.remove(object_id)
                p.removeBody(object_id)
                total_picked += 1
        # any object is picked is fine
        if total_picked == 1:
            reward += 10
        return reward

    def check_if(self, objectorn, gripper_yaw):
        if abs(math.degrees(objectorn[2])) > 46 and abs(gripper_yaw) > 80:
            if_yaw_in = True
        elif abs(math.degrees(objectorn[2])) < 44 and abs(gripper_yaw) < 10:
            if_yaw_in = True
        elif 44 < abs(math.degrees(objectorn[2])) < 46:
            if_yaw_in = True
        else:
            if_yaw_in = False
        return if_yaw_in

    def get_robotid(self):
        return self.robot_id, self.gripper_urdf, self.table_id
