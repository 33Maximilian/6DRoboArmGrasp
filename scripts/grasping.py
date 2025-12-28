#!/usr/bin/env python3
"""抓取执行节点，用于将跟踪输出桥接到FollowJointTrajectory。"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped, PoseStamped, QuaternionStamped
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import yaml
from ament_index_python.packages import PackageNotFoundError
from ament_index_python.packages import get_package_share_directory


@dataclass
class GraspCandidate:
    """用于保存同步抓取样本的容器。"""

    position: PointStamped
    rotation: QuaternionStamped
    width: float
    score: float


class GraspingNode(Node):
    """订阅AnyGrasp话题并通过轨迹动作控制机械臂。"""

    def __init__(self) -> None:
        super().__init__('grasping_node')
        self.cb_group = ReentrantCallbackGroup()

        self.manipulator_variant = self.declare_parameter('manipulator_variant', 'omx_f').value
        self.param_override_path = self.declare_parameter('official_params_file', '').value
        official_defaults = self._load_official_defaults()

        # 核心参数，镜像joint_trajectory_executor的默认值
        self.declare_parameter('joint_names', official_defaults['joint_names'])
        self.declare_parameter(
            'action_topic', official_defaults['action_topic']
        )
        self.declare_parameter('joint_states_topic', official_defaults['joint_states_topic'])
        self.declare_parameter('trajectory_duration', official_defaults['trajectory_duration'])
        self.declare_parameter('trajectory_samples', 100)
        self.declare_parameter('goal_time_tolerance', 0.0)
        self.declare_parameter('control_rate', 5.0)
        self.declare_parameter('epsilon', official_defaults['epsilon'])

        # 抓取特定参数
        self.declare_parameter('sync_window', 0.05)
        self.declare_parameter('score_threshold', 0.4)
        self.declare_parameter('queue_size', 5)
        self.declare_parameter('tf_timeout', 0.5)
        self.declare_parameter('ik_timeout', 1.0)
        self.declare_parameter('group_name', 'open_manipulator')
        self.declare_parameter('ik_service_name', '/compute_ik')
        self.declare_parameter('base_frame', 'world')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('tf_required', True)
        self.declare_parameter('bypass_ik', False)
        self.declare_parameter('fallback_joint_positions', [])

        self.joint_names: List[str] = (
            self.get_parameter('joint_names').get_parameter_value().string_array_value
        )
        self.action_topic = self.get_parameter('action_topic').value
        self.joint_states_topic = self.get_parameter('joint_states_topic').value
        self.duration = self.get_parameter('trajectory_duration').value
        self.num_points = int(self.get_parameter('trajectory_samples').value)
        self.goal_time_tolerance_sec = self.get_parameter('goal_time_tolerance').value
        self.epsilon = self.get_parameter('epsilon').value
        self.score_threshold = self.get_parameter('score_threshold').value
        self.sync_window = self.get_parameter('sync_window').value
        self.queue_size = int(self.get_parameter('queue_size').value)
        self.tf_timeout = self.get_parameter('tf_timeout').value
        self.ik_timeout = self.get_parameter('ik_timeout').value
        self.group_name = self.get_parameter('group_name').value
        self.ik_service_name = self.get_parameter('ik_service_name').value
        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.tf_required = self.get_parameter('tf_required').value
        self.bypass_ik = self.get_parameter('bypass_ik').value
        self.fallback_joint_positions = list(
            self.get_parameter('fallback_joint_positions').value
        )

        if self.official_params_path:
            self.get_logger().info(f'使用官方参数文件默认值: {self.official_params_path}')
        else:
            self.get_logger().warn('官方参数文件解析失败，使用脚本内置默认值')

        # 运行时状态
        self.current_joint_state: Optional[JointState] = None
        self.current_positions: Optional[List[float]] = None
        self.latest_position: Optional[PointStamped] = None
        self.latest_rotation: Optional[QuaternionStamped] = None
        self.latest_width: Optional[float] = None
        self.latest_score: Optional[float] = None
        self.processing = False
        self.goal_handle = None
        self.grasp_queue: Deque[GraspCandidate] = deque(maxlen=self.queue_size)

        # ROS接口
        self.action_client = ActionClient(
            self, FollowJointTrajectory, self.action_topic, callback_group=self.cb_group
        )
        self.joint_state_sub = self.create_subscription(
            JointState, self.joint_states_topic, self.joint_state_callback, 10
        )
        self.position_sub = self.create_subscription(
            PointStamped, '/grasp/position', self.position_callback, 10
        )
        self.rotation_sub = self.create_subscription(
            QuaternionStamped, '/grasp/rotation', self.rotation_callback, 10
        )
        self.width_sub = self.create_subscription(
            Float64, '/grasp/width', self.width_callback, 10
        )
        self.score_sub = self.create_subscription(
            Float64, '/grasp/score', self.score_callback, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.ik_client = self.create_client(
            GetPositionIK, self.ik_service_name, callback_group=self.cb_group
        )

        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                f'无法在 {self.action_topic} 上找到 FollowJointTrajectory 动作'
            )
            raise RuntimeError('FollowJointTrajectory 动作不可用')
        self.get_logger().info('FollowJointTrajectory 动作就绪')

        if not self.ik_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                f'IK 服务 {self.ik_service_name} 尚不可用；将继续尝试'
            )

        self.control_timer = self.create_timer(
            1.0 / max(self.get_parameter('control_rate').value, 1e-3),
            self.process_queue,
        )
        self.get_logger().info('抓取节点初始化完成')

    # --- 订阅回调函数 -------------------------------------------------
    def joint_state_callback(self, msg: JointState) -> None:
        if not set(self.joint_names).issubset(set(msg.name)):
            return
        self.current_joint_state = msg
        self.current_positions = [msg.position[msg.name.index(j)] for j in self.joint_names]

    def position_callback(self, msg: PointStamped) -> None:
        self.latest_position = msg
        self.try_queue_grasp()

    def rotation_callback(self, msg: QuaternionStamped) -> None:
        self.latest_rotation = msg
        self.try_queue_grasp()

    def width_callback(self, msg: Float64) -> None:
        self.latest_width = msg.data
        self.try_queue_grasp()

    def score_callback(self, msg: Float64) -> None:
        self.latest_score = msg.data
        self.try_queue_grasp()

    # --- 抓取聚合 -----------------------------------------------------
    def try_queue_grasp(self) -> None:
        if None in (
            self.latest_position,
            self.latest_rotation,
            self.latest_width,
            self.latest_score,
        ):
            return

        pos_time = self._stamp_to_float(self.latest_position.header.stamp)
        rot_time = self._stamp_to_float(self.latest_rotation.header.stamp)
        if abs(pos_time - rot_time) > self.sync_window:
            return  # 等待数据时间接近

        candidate = GraspCandidate(
            position=self.latest_position,
            rotation=self.latest_rotation,
            width=self.latest_width,
            score=self.latest_score,
        )
        if candidate.score < self.score_threshold:
            self.get_logger().debug(
                f'丢弃低分抓取样本 {candidate.score:.2f} (阈值 {self.score_threshold})'
            )
            return

        if len(self.grasp_queue) == self.grasp_queue.maxlen:
            self.grasp_queue.pop()  # 丢弃最旧数据以保留最新数据
        self.grasp_queue.appendleft(candidate)
        self.get_logger().info(
            f'已加入新抓取队列 (分数={candidate.score:.3f}, 宽度={candidate.width:.3f} m)'
        )

    # --- 处理循环 -------------------------------------------------------
    def process_queue(self) -> None:
        if self.processing or not self.grasp_queue:
            return
        if self.current_joint_state is None:
            self.get_logger().warn_throttle(
                5.0, '等待关节状态更新以执行抓取'
            )
            return
        candidate = self.grasp_queue.pop()
        self.processing = True
        self.execute_candidate(candidate)

    def execute_candidate(self, candidate: GraspCandidate) -> None:
        if self.bypass_ik:
            joint_targets = self._resolve_fallback_targets()
            if joint_targets is None:
                self.processing = False
                return
            self.send_trajectory_goal(joint_targets, source='fallback')
            return

        if self.ik_client.service_is_ready() is False:
            joint_targets = self._resolve_fallback_targets()
            if joint_targets is not None:
                self.get_logger().warn_throttle(
                    5.0,
                    'IK服务不可用，使用备用关节位置执行轨迹',
                )
                self.send_trajectory_goal(joint_targets, source='fallback')
                return

            self.get_logger().warn_throttle(
                5.0,
                f'IK服务 {self.ik_service_name} 仍不可用；重新加入抓取队列',
            )
            self.processing = False
            self.grasp_queue.append(candidate)
            return

        pose = PoseStamped()
        pose.header = candidate.position.header
        pose.pose.position = candidate.position.point
        pose.pose.orientation = candidate.rotation.quaternion

        transformed_pose = self.transform_pose_to_base(pose)
        if transformed_pose is None:
            self.processing = False
            return

        self.request_ik(transformed_pose, candidate)

    def transform_pose_to_base(self, pose: PoseStamped) -> Optional[PoseStamped]:
        if not self.tf_required:
            pose.header.frame_id = self.base_frame
            return pose

        try:
            target_time = rclpy.time.Time.from_msg(pose.header.stamp)
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                pose.header.frame_id or self.camera_frame,
                target_time,
                timeout=Duration(seconds=self.tf_timeout),
            )
            return do_transform_pose(pose, transform)
        except Exception as exc:  # TF不可用，标定仍在进行中
            self.get_logger().warn(
                f'TF查找失败 ({pose.header.frame_id}->{self.base_frame}): {exc}'
            )
            return None

    def request_ik(self, pose: PoseStamped, candidate: GraspCandidate) -> None:
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.pose_stamped = pose
        req.ik_request.attempts = 5
        req.ik_request.timeout = Duration(seconds=self.ik_timeout).to_msg()
        if self.current_joint_state:
            req.ik_request.robot_state.joint_state = self.current_joint_state

        future = self.ik_client.call_async(req)
        future.add_done_callback(lambda f: self.handle_ik_response(f, candidate))

    def handle_ik_response(self, future, candidate: GraspCandidate) -> None:
        if future.cancelled():
            self.get_logger().warn('IK请求已取消')
            self.processing = False
            return
        if future.exception() is not None:
            self.get_logger().error(f'IK请求失败: {future.exception()}')
            self.processing = False
            return

        response = future.result()
        error_val = response.error_code.val
        if error_val != 1:  # 成功
            self.get_logger().warn(f'IK求解器返回错误代码 {error_val}')
            self.processing = False
            return

        joint_targets = self.extract_joint_targets(response.solution.joint_state)
        if joint_targets is None:
            self.processing = False
            return

        self.send_trajectory_goal(joint_targets, source='ik')

    def extract_joint_targets(self, joint_state: JointState) -> Optional[List[float]]:
        if not set(self.joint_names).issubset(set(joint_state.name)):
            self.get_logger().error('IK解缺少必要的关节')
            return None
        return [joint_state.position[joint_state.name.index(j)] for j in self.joint_names]

    def _resolve_fallback_targets(self) -> Optional[List[float]]:
        if self.fallback_joint_positions:
            if len(self.fallback_joint_positions) != len(self.joint_names):
                self.get_logger().error(
                    'fallback_joint_positions 长度与 joint_names 不一致'
                )
                return None
            return [float(val) for val in self.fallback_joint_positions]
        if self.current_positions is not None:
            return list(self.current_positions)
        self.get_logger().warn('没有可用的备用关节位置，跳过抓取')
        return None

    def send_trajectory_goal(self, joint_targets: List[float], source: str) -> None:
        traj = self.create_smooth_trajectory(self.current_positions, joint_targets)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        goal.goal_time_tolerance = Duration(
            seconds=self.goal_time_tolerance_sec
        ).to_msg()

        send_future = self.action_client.send_goal_async(
            goal, feedback_callback=self.feedback_callback
        )
        send_future.add_done_callback(self.goal_response_callback)
        self.get_logger().info(f'已将轨迹目标发送到机械臂控制器 (source={source})')

    # --- 动作回调 -------------------------------------------------------
    def goal_response_callback(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('控制器拒绝了轨迹目标')
            self.processing = False
            return

        self.goal_handle = goal_handle
        self.get_logger().info('轨迹目标已被接受')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future) -> None:
        self.processing = False
        if future.exception() is not None:
            self.get_logger().error(f'执行轨迹失败: {future.exception()}')
            return
        result = future.result().result
        self.get_logger().info(f'目标结果已接收 (错误代码={result.error_code})')

    def feedback_callback(self, feedback_msg) -> None:
        actual = feedback_msg.feedback.actual.positions
        target = feedback_msg.feedback.desired.positions
        error = max(abs(a - b) for a, b in zip(actual, target))
        if error < self.epsilon:
            self.get_logger().debug('接近抓取目标')

    # --- 工具函数 --------------------------------------------------------------
    def create_smooth_trajectory(
        self, start_pos: Optional[List[float]], end_pos: List[float]
    ) -> JointTrajectory:
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        if start_pos is None:
            start_pos = end_pos

        times = np.linspace(0, self.duration, self.num_points)
        for t in times:
            point = JointTrajectoryPoint()
            t_norm = t / self.duration if self.duration > 0 else 1.0
            t2 = t_norm * t_norm
            t3 = t2 * t_norm
            t4 = t3 * t_norm
            t5 = t4 * t_norm

            pos_coeff = 10 * t3 - 15 * t4 + 6 * t5
            vel_coeff = (30 * t2 - 60 * t3 + 30 * t4) / max(self.duration, 1e-6)
            acc_coeff = (60 * t_norm - 180 * t2 + 120 * t3) / max(
                self.duration * self.duration, 1e-6
            )

            positions = []
            velocities = []
            accelerations = []
            for idx, joint in enumerate(self.joint_names):
                s = start_pos[idx]
                e = end_pos[idx]
                delta = e - s
                positions.append(s + delta * pos_coeff)
                velocities.append(delta * vel_coeff)
                accelerations.append(delta * acc_coeff)

            point.positions = positions
            point.velocities = velocities
            point.accelerations = accelerations
            point.time_from_start = DurationMsg(sec=int(t), nanosec=int((t % 1) * 1e9))
            traj.points.append(point)
        return traj

    @staticmethod
    def _stamp_to_float(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _load_official_defaults(self) -> dict:
        defaults = {
            'joint_names': [''],
            'action_topic': '/arm_controller/follow_joint_trajectory',
            'joint_states_topic': '/joint_states',
            'trajectory_duration': 10.0,
            'epsilon': 0.01,
        }

        self.official_params_path = self._resolve_official_params_path()
        path = self.official_params_path
        if not path or not os.path.exists(path):
            return defaults

        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = yaml.safe_load(file) or {}
        except Exception as exc:
            self.get_logger().warn(f'无法读取官方参数文件 {path}: {exc}')
            return defaults

        params = (
            content.get('joint_trajectory_executor', {})
            .get('ros__parameters', {})
        )
        defaults['joint_names'] = params.get('joint_names', defaults['joint_names'])
        defaults['action_topic'] = params.get('action_topic', defaults['action_topic'])
        defaults['joint_states_topic'] = params.get(
            'joint_states_topic', defaults['joint_states_topic']
        )
        defaults['trajectory_duration'] = params.get(
            'duration', defaults['trajectory_duration']
        )
        defaults['epsilon'] = params.get('epsilon', defaults['epsilon'])
        return defaults

    def _resolve_official_params_path(self) -> Optional[str]:
        if self.param_override_path:
            override_path = os.path.abspath(os.path.expanduser(self.param_override_path))
            return override_path

        try:
            package_path = get_package_share_directory('open_manipulator_bringup')
        except PackageNotFoundError:
            self.get_logger().warn('找不到 open_manipulator_bringup 包，无法定位官方参数文件')
            return None

        candidate = os.path.join(
            package_path,
            'config',
            self.manipulator_variant,
            'initial_positions.yaml',
        )
        return candidate


def main() -> None:
    rclpy.init()
    node = GraspingNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
