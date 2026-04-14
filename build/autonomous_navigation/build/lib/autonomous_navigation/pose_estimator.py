import math

import rclpy
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener

from .config import euler_from_quaternion, normalize_angle


class PoseEstimator:
    def __init__(self, node):
        self.node = node
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)

        # Pose state (map frame)
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.pose_source = "none"
        self.initial_pose_received = False

        # Raw odometry
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.odom_received = False

        # Manual odom->map anchor from user-provided initial pose
        self.manual_anchor_ready = False
        self.anchor_rot = 0.0
        self.anchor_tx = 0.0
        self.anchor_ty = 0.0

    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion(q)
        self.odom_x = pos.x
        self.odom_y = pos.y
        self.odom_yaw = yaw
        self.odom_received = True

    def set_initial_pose_estimate(self, x: float, y: float, yaw: float):
        self.current_x = x
        self.current_y = y
        self.current_yaw = yaw
        self.initial_pose_received = True

    def set_manual_anchor_from_initial_pose(self, init_x: float, init_y: float, init_yaw: float) -> bool:
        if not self.odom_received:
            return False

        self.anchor_rot = normalize_angle(init_yaw - self.odom_yaw)
        cos_r = math.cos(self.anchor_rot)
        sin_r = math.sin(self.anchor_rot)
        self.anchor_tx = init_x - (cos_r * self.odom_x - sin_r * self.odom_y)
        self.anchor_ty = init_y - (sin_r * self.odom_x + cos_r * self.odom_y)
        self.manual_anchor_ready = True
        return True

    def update_pose_from_manual_anchor(self) -> bool:
        if not (self.manual_anchor_ready and self.odom_received):
            return False

        cos_r = math.cos(self.anchor_rot)
        sin_r = math.sin(self.anchor_rot)
        self.current_x = cos_r * self.odom_x - sin_r * self.odom_y + self.anchor_tx
        self.current_y = sin_r * self.odom_x + cos_r * self.odom_y + self.anchor_ty
        self.current_yaw = normalize_angle(self.odom_yaw + self.anchor_rot)
        self.pose_source = "manual_anchor"
        return True

    def get_robot_pose(self) -> bool:
        if self.update_pose_from_manual_anchor():
            return True

        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            self.current_x = t.transform.translation.x
            self.current_y = t.transform.translation.y
            q = t.transform.rotation
            _, _, self.current_yaw = euler_from_quaternion(q)
            self.pose_source = "tf"
            return True
        except Exception:
            return False
