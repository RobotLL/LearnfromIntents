import pyrealsense2 as rs
import numpy as np
import json
import math
import matplotlib.pyplot as plt
import cv2

# 170mm
class L515:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        print("[INFO] start streaming...")
        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", depth_scale)

        intr = (
            profile.get_stream(rs.stream.depth)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        print("width is: ", intr.width)
        print("height is: ", intr.height)
        print("ppx is: ", intr.ppx)
        print("ppy is: ", intr.ppy)
        print("fx is: ", intr.fx)
        print("fy is: ", intr.fy)
        HFOV = math.degrees(2 * math.atan(intr.width / (intr.fx + intr.fy)))
        print("HFOV is", HFOV)
        VFOV = math.degrees(2 * math.atan(intr.height / (intr.fx + intr.fy)))
        print("VFOV is", VFOV)
        self.point_cloud = rs.pointcloud()

    def get_verts(self):

        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        decimation = rs.decimation_filter(4)
        depth_frame = decimation.process(depth_frame)
        points = self.point_cloud.calculate(depth_frame)
        verts = (
            np.asanyarray(points.get_vertices()).view(np.float32).reshape(120, 160, 3)
        )  # xyz
        return verts
    
    def get_full_dpi_verts(self):

        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        points = self.point_cloud.calculate(depth_frame)
        full_dpi_verts = (
            np.asanyarray(points.get_vertices()).view(np.float32).reshape(480, 640, 3)
        )  # xyz
        return full_dpi_verts

    def stop_streaming(self):
        self.pipeline.stop()


