import time
import time
import socket
import urx
from real_gripper import Robotiq_Two_Finger_Gripper
import pybullet as p

# from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import math
import os
import math3d as m3d
#from util import gripper_orn_to_world

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# crad_00_center
Camera_x_offset = 0.165
Camera_y_offset = 0
Camera_z_offset = 0


class RealRobot:
    def __init__(self, table=True, chuan=False):
        # Real hardware
        self.robot = urx.Robot("192.168.1.102", use_rt=True)
        self.robotiq_gripper = Robotiq_Two_Finger_Gripper(self.robot)
        self.robotiq_gripper.gripper_action_full(210, 1, 1)
        self.X_OFFSET = 0.7
        self.Y_OFFSET = 0.125
        self.Z_OFFSET = 0.5
        if table is True:
            self.X_OFFSET = 0.7 + 0.05/2
            self.Y_OFFSET = 0.125
            self.Z_OFFSET = 0.5 - 0.005
        if chuan is True:
            self.X_OFFSET = 0.7 + 0.025
            self.Y_OFFSET = 0.125
            self.Z_OFFSET = 0.5 + 0.005

    def close_gripper(self, length=210, speed=10):
        self.robotiq_gripper.gripper_action_full(length, speed, 10)

    def open_gripper(self, length=175):
        self.robotiq_gripper.gripper_action_full(
            length, 100, 100
        )  # 2.5cm gripper open angle

    def go_r_home(self):
        self.set_tcp(0, 0, 0, 0, 0, 0)
        self.move_world(self.X_OFFSET, self.Y_OFFSET, self.Z_OFFSET-0.1, math.pi, 0, 0, 0.3, 0.3)

    def go_c_home(self):
        self.set_tcp(0, 0, 0, 0, 0, 0)
        self.move_world(self.X_OFFSET-Camera_x_offset, self.Y_OFFSET-Camera_y_offset, self.Z_OFFSET, math.pi, 0, 0, 0.3, 0.3)

    def go_chuan_home(self):
        self.set_tcp(0, 0, 0, 0, 0, 0)
        self.move_world(self.X_OFFSET-Camera_x_offset, self.Y_OFFSET-Camera_y_offset, self.Z_OFFSET, math.pi, 0, 0, 0.3, 0.3)

    def move_joint(self, home_position, acc=0.1, vel=0.1):
        Hong_joint0 = math.radians(home_position[0])
        Hong_joint1 = math.radians(home_position[1])
        Hong_joint2 = math.radians(home_position[2])
        Hong_joint3 = math.radians(home_position[3])
        Hong_joint4 = math.radians(home_position[4])
        Hong_joint5 = math.radians(home_position[5])
        self.robot.movej(
            (
                Hong_joint0,
                Hong_joint1,
                Hong_joint2,
                Hong_joint3,
                Hong_joint4,
                Hong_joint5,
            ),
            acc,
            vel,
        )

    def set_tcp(self, x, y, z, rx, ry, rz):
        self.robot.set_tcp((x, y, z, rx, ry, rz))
        time.sleep(1)

    def get_tool_position(self):
        tool_pose = self.robot.getl()
        x = tool_pose[0] - self.X_OFFSET
        y = tool_pose[1] - self.Y_OFFSET
        z = tool_pose[2]
        return (x, y, z)

    def cal_furure(self, targetxyz):
        tool_pose = self.robot.getl()
        x = tool_pose[0] - self.X_OFFSET
        y = tool_pose[1] - self.Y_OFFSET
        z = tool_pose[2]
        rel_x = targetxyz[0] - x
        rel_y = targetxyz[1] - y
        rel_z = targetxyz[2] - z
        return rel_x, rel_y, rel_z

    def move_with_ori(
        self, targetxyz, acc=0.05, vel=0.05, wait=True, relative=True
    ):
        tool_pose = self.robot.getl()
        x = tool_pose[0] - self.X_OFFSET
        y = tool_pose[1] - self.Y_OFFSET
        z = tool_pose[2]
        rela_x = targetxyz[0] - x
        rela_y = targetxyz[1] - y
        rela_z = targetxyz[2] - z
        self.robot.movel((rela_x, rela_y, rela_z, 0, 0, 0), acc, vel, wait, relative)

    def move_world(
        self, wx, wy, wz, rx, ry, rz, acc=0.1, vel=0.1, wait=True, relative=True
    ):
        self.robot.movel((wx, wy, wz, rx, ry, rz), acc, vel)

    def move_tool_z(self, z, acc=0.01, vel=0.01):
        self.robot.translate((0, 0, z), acc, vel)

    def set_gripper_ori(self, roll, pitch, yaw, acc=0.1, vel=0.1, wait=True):
        rot_acc = acc
        rot_vel = vel
        move = m3d.Transform((0, 0, 0, 0, 0, -yaw))
        self.robot.add_pose_tool(
            move, acc=rot_acc, vel=rot_vel, wait=wait, command="movel", threshold=None
        )
        move = m3d.Transform((0, 0, 0, roll, 0, 0))
        self.robot.add_pose_tool(
            move, acc=rot_acc, vel=rot_vel, wait=wait, command="movel", threshold=None
        )
        move = m3d.Transform((0, 0, 0, 0, pitch, 0))
        self.robot.add_pose_tool(
            move, acc=rot_acc, vel=rot_vel, wait=wait, command="movel", threshold=None
        )

    def resetFT300Sensor(self, tcp_host_ip="192.168.1.102"):
        HOST = tcp_host_ip
        PORT = 63351
        self.serialFT300Sensor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serialFT300Sensor.connect((HOST, PORT))

    def getFT300SensorData(self):
        while True:
            data = (
                str(self.serialFT300Sensor.recv(1024), "utf-8")
                .replace("(", "")
                .replace(")", "")
                .split(",")
            )
            try:
                data = [float(x) for x in data]
                if len(data) == 6:
                    break
            except:
                pass
        return data

    def get_rob(self):
        pass
        return self.robot

