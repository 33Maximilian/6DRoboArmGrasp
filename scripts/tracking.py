#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, QuaternionStamped
from std_msgs.msg import Float64
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'anygrasp_sdk', 'grasp_tracking'))
from tracker import AnyGraspTracker


class RealSenseAdapter:
    """点云获取"""
    def __init__(self, width=640, height=480, fps=30, clip_distance=1.5):
        self.clip_distance = clip_distance
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.align = self.pc = self.color_intrin = None
        
    def start(self):
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()
        depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        print(f"RealSense 已就绪 (depth_scale={depth_scale:.4f}, clip_distance={self.clip_distance}m)\n")
        
    def stop(self):
        if self.pipeline:
            self.pipeline.stop()
    
    def get_pointcloud_data(self):
        """获取对齐的点云和颜色数据"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None
        
        if self.color_intrin is None:
            self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        
        # 生成点云
        self.pc.map_to(color_frame)
        points_obj = self.pc.calculate(depth_frame)
        verts = np.asanyarray(points_obj.get_vertices()).view(np.float32).reshape(-1, 3)
        texcoords = np.asanyarray(points_obj.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
        
        # 深度滤波
        mask = verts[:, 2] < self.clip_distance
        verts = verts[mask]
        texcoords = texcoords[mask]
        
        # 颜色采样
        color_image = np.asanyarray(color_frame.get_data())
        h, w = color_image.shape[:2]
        u = np.clip((texcoords[:, 0] * w).astype(np.int32), 0, w - 1)
        v = np.clip((texcoords[:, 1] * h).astype(np.int32), 0, h - 1)
        
        colors_bgr = color_image[v, u]
        colors_rgb = cv2.cvtColor(colors_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
        return verts, colors_rgb.astype(np.float32) / 255.0


class GraspTrackerNode(Node):
    """ROS2 抓取追踪节点"""
    def __init__(self, adapter, tracker, args):
        super().__init__('grasp_tracker_node')
        
        self.adapter = adapter
        self.tracker = tracker
        self.args = args
        
        # 发布者
        self.pos_pub = self.create_publisher(PointStamped, '/grasp/position', 10)
        self.rot_pub = self.create_publisher(QuaternionStamped, '/grasp/rotation', 10)
        self.width_pub = self.create_publisher(Float64, '/grasp/width', 10)
        self.score_pub = self.create_publisher(Float64, '/grasp/score', 10)
        
        # 追踪状态
        self.frame_count = 0
        self.prev_count = 0
        self.prev_score = 0.5
        self.last_redetect = -999
        self.grasp_ids = np.array([0])
        
        # 可视化
        self.vis = None
        if args.debug:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name='AnyGrasp Tracking', height=720, width=1280)
        
        self.create_timer(0.033, self.timer_callback)
        self.get_logger().info('ROS2 发布器启动: /grasp/{position,rotation,width,score}')
    
    def timer_callback(self):
        """主循环: 追踪 + 发布 + 可视化"""
        points, colors = self.adapter.get_pointcloud_data()
        if points is None:
            return
        
        self.frame_count += 1
        target_gg, curr_gg, target_ids, _ = self.tracker.update(points, colors, self.grasp_ids)
        
        # 质量监测
        can_redetect = (self.frame_count - self.last_redetect) >= self.args.redetect_cooldown
        force_redetect = False
        
        if self.frame_count > 1 and len(target_gg) > 0 and can_redetect:
            loss_ratio = (self.prev_count - len(target_gg)) / max(self.prev_count, 1)
            if loss_ratio > self.args.track_loss_threshold:
                force_redetect = True
            elif self.prev_score > 0:
                drop_ratio = (self.prev_score - target_gg.scores.mean()) / self.prev_score
                if drop_ratio > self.args.score_drop_threshold:
                    force_redetect = True
        
        if self.frame_count == 1 or force_redetect:
            self.last_redetect = self.frame_count
            self.grasp_ids = np.arange(len(curr_gg))[::self.args.grasp_stride]
            target_gg = curr_gg[self.grasp_ids]
            self.prev_count = len(target_gg)
            if len(target_gg) > 0:
                self.prev_score = target_gg.scores.mean()
            score_range = f"[{target_gg.scores.min():.3f}, {target_gg.scores.max():.3f}]" if len(target_gg) > 0 else "[0, 0]"
            msg = f"{'首帧' if self.frame_count == 1 else '重检'}: {len(curr_gg)}→{len(target_gg)} {score_range}"
            self.get_logger().info(msg)
        else:
            self.grasp_ids = np.array(target_ids) if target_ids is not None else self.grasp_ids
            if len(target_gg) > 0:
                self.prev_count = len(target_gg)
                self.prev_score = target_gg.scores.mean()
            if self.frame_count % 30 == 1:
                score_range = f"[{target_gg.scores.min():.3f}, {target_gg.scores.max():.3f}]" if len(target_gg) > 0 else "[0, 0]"
                self.get_logger().info(f"F{self.frame_count}: {len(curr_gg)}→{len(target_gg)} {score_range}")
        
        if len(target_gg) > 0:
            self.prev_count = len(target_gg)
            self.prev_score = target_gg.scores.mean()
            self.publish_grasp(target_gg[0])
            
            if self.args.debug and self.vis:
                self.visualize(points, colors, target_gg)
    
    def publish_grasp(self, grasp):
        try:
            if not rclpy.ok():
                return
            
            now = self.get_clock().now().to_msg()
            frame_id = 'camera_link'
            
            # 位置
            pos = PointStamped()
            pos.header.stamp, pos.header.frame_id = now, frame_id
            pos.point.x, pos.point.y, pos.point.z = map(float, grasp.translation)
            self.pos_pub.publish(pos)
            
            # 姿态
            rot = Rotation.from_matrix(grasp.rotation_matrix).as_quat()
            quat = QuaternionStamped()
            quat.header.stamp, quat.header.frame_id = now, frame_id
            quat.quaternion.x, quat.quaternion.y, quat.quaternion.z, quat.quaternion.w = map(float, rot)
            self.rot_pub.publish(quat)
            
            # 宽度和评分
            self.width_pub.publish(Float64(data=float(grasp.width)))
            self.score_pub.publish(Float64(data=float(grasp.score)))
        except Exception as e:
            self.get_logger().error(f'发布失败: {e}')
    
    def visualize(self, points, colors, target_gg):
        trans = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        cloud.transform(trans)
        
        self.vis.clear_geometries()
        self.vis.add_geometry(cloud)
        for g in target_gg.to_open3d_geometry_list():
            self.vis.add_geometry(g.transform(trans) or g)
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def destroy_node(self):
        if self.vis:
            self.vis.destroy_window()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', required=True, help='追踪模型路径')
    parser.add_argument('--clip-distance', type=float, default=1.5, help='深度裁剪距离 (m)')
    parser.add_argument('--filter', default='oneeuro', help='滤波器类型')
    parser.add_argument('--debug', action='store_true', help='启用可视化')
    parser.add_argument('--grasp-stride', type=int, default=6, help='检测采样间隔')
    parser.add_argument('--redetect-cooldown', type=int, default=30, help='重检冷却帧数')
    parser.add_argument('--score-drop-threshold', type=float, default=0.4, help='评分下降阈值')
    parser.add_argument('--track-loss-threshold', type=float, default=0.5, help='追踪丢失阈值')
    args = parser.parse_args()
    
    # 初始化
    print("初始化追踪模型...")
    tracker = AnyGraspTracker(type('Config', (), {'checkpoint_path': args.checkpoint_path, 'filter': args.filter, 'debug': args.debug}))
    tracker.load_net()
    
    adapter = RealSenseAdapter(clip_distance=args.clip_distance)
    adapter.start()
    
    # ROS2 节点
    rclpy.init()
    node = GraspTrackerNode(adapter, tracker, args)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        adapter.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
