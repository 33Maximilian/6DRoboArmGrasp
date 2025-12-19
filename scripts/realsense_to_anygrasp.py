#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'anygrasp_sdk', 'grasp_tracking'))
from tracker import AnyGraspTracker
from graspnetAPI import GraspGroup


class RealSenseToAnyGrasp:
    def __init__(self, width=640, height=480, fps=30, clip_distance=1.5):
        self.width = width
        self.height = height
        self.fps = fps
        self.clip_distance = clip_distance
    
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        self.profile = None
        self.align = None
        self.pc = None
        self.depth_scale = None
        self.color_intrin = None
        
    def start(self):
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        print(f"RealSense: {self.width}×{self.height}, scale={self.depth_scale:.4f}, clip={self.clip_distance}m\n")
        
    def stop(self):
        if self.pipeline:
            self.pipeline.stop()
    
    def get_pointcloud_data(self):
        """获取点云和颜色数据 (N,3) 米, (N,3) RGB [0,1]"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            return None, None
        
        if self.color_intrin is None:
            self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            print(f"相机内参: fx={self.color_intrin.fx:.2f}, fy={self.color_intrin.fy:.2f}, "
                  f"cx={self.color_intrin.ppx:.2f}, cy={self.color_intrin.ppy:.2f}\n")
        
        color_image = np.asanyarray(color_frame.get_data())
        
        # 生成点云和纹理坐标
        self.pc.map_to(color_frame)
        points_rs = self.pc.calculate(aligned_depth_frame)
        verts = np.asanyarray(points_rs.get_vertices()).view(np.float32).reshape(-1, 3)
        texcoords = np.asanyarray(points_rs.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
        
        # 深度滤波
        mask_depth = verts[:, 2] < self.clip_distance
        verts_filtered = verts[mask_depth]
        texcoords_filtered = texcoords[mask_depth]
        
        # 通过纹理坐标采样颜色
        h, w = color_image.shape[:2]
        v_tex = np.clip((texcoords_filtered[:, 0] * w).astype(np.int32), 0, w - 1)
        u_tex = np.clip((texcoords_filtered[:, 1] * h).astype(np.int32), 0, h - 1)
        
        colors_bgr = color_image[u_tex, v_tex]
        colors_rgb = cv2.cvtColor(colors_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
        colors = colors_rgb.astype(np.float32) / 255.0
        
        return verts_filtered, colors
    
class AnyGraspConfig:
    def __init__(self, checkpoint_path, filter_type='oneeuro', debug=True):
        self.checkpoint_path = checkpoint_path
        self.filter = filter_type
        self.debug = debug


def run_tracking():
    parser = argparse.ArgumentParser(description='RealSense + AnyGrasp tracking')
    parser.add_argument('--checkpoint-path', required=True, help='Model checkpoint path')
    parser.add_argument('--clip-distance', type=float, default=1.5, help='Clipping distance (m)')
    parser.add_argument('--filter', type=str, default='oneeuro', choices=['oneeuro', 'kalman', 'none'])
    parser.add_argument('--debug', action='store_true', help='Enable visualization')
    parser.add_argument('--grasp-stride', type=int, default=6, help='Stride for grasp selection')
    parser.add_argument('--redetect-cooldown', type=int, default=30, help='Minimum frames between redetections')
    parser.add_argument('--score-drop-threshold', type=float, default=0.4, help='Score drop ratio to trigger redetect')
    parser.add_argument('--track-loss-threshold', type=float, default=0.5, help='Track loss ratio to trigger redetect')
    args = parser.parse_args()
    
    print("=== 初始化 AnyGrasp Tracker ===")
    cfgs = AnyGraspConfig(args.checkpoint_path, args.filter, args.debug)
    anygrasp_tracker = AnyGraspTracker(cfgs)
    anygrasp_tracker.load_net()
    
    print("=== 初始化 RealSense ===")
    adapter = RealSenseToAnyGrasp(clip_distance=args.clip_distance)
    adapter.start()
    
    grasp_ids = [0]
    frame_count = 0
    prev_target_count = 0
    prev_avg_score = 0.5
    last_redetect_frame = -999
    
    vis = None
    if args.debug:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='AnyGrasp Tracking', height=720, width=1280)
    
    print("=== 开始实时追踪 ===")
    
    try:
        while True:
            points, colors = adapter.get_pointcloud_data()
            
            if points is None:
                continue
            
            frame_count += 1
            
            # 调用 tracker.update() 获取抓取
            target_gg, curr_gg, target_grasp_ids, corres_preds = \
                anygrasp_tracker.update(points, colors, grasp_ids)
            
            # 质量监测
            force_redetect = False
            redetect_reason = ""
            
            can_redetect = (frame_count - last_redetect_frame) >= args.redetect_cooldown
            
            if frame_count > 1 and len(target_gg) > 0 and can_redetect:
                # 计算追踪丢失率
                track_loss_ratio = (prev_target_count - len(target_gg)) / max(prev_target_count, 1)
                if track_loss_ratio > args.track_loss_threshold:
                    force_redetect = True
                    redetect_reason = f"目标丢失{track_loss_ratio*100:.1f}%"
                
                # 计算得分下降率
                curr_avg_score = target_gg.scores.mean()
                if prev_avg_score > 0 and curr_avg_score < prev_avg_score:
                    score_drop_ratio = (prev_avg_score - curr_avg_score) / prev_avg_score
                    if score_drop_ratio > args.score_drop_threshold:
                        force_redetect = True
                        redetect_reason = f"得分下降{score_drop_ratio*100:.1f}%"
            
            if force_redetect:
                last_redetect_frame = frame_count
            
            if frame_count == 1 or force_redetect:
                grasp_ids = np.arange(len(curr_gg))[::args.grasp_stride]
                target_gg = curr_gg[grasp_ids]
                prev_target_count = len(target_gg)
                if len(target_gg) > 0:
                    prev_avg_score = target_gg.scores.mean()
                
                score_min, score_max = (target_gg.scores.min(), target_gg.scores.max()) if len(target_gg) > 0 else (0, 0)
                if frame_count == 1:
                    print(f"第一帧: 检测 {len(curr_gg)} 抓取, 追踪 {len(target_gg)} 抓取，得分: [{score_min:.3f}, {score_max:.3f}]")
                else:
                    print(f"重检({redetect_reason}): 检测 {len(curr_gg)} 抓取, 追踪 {len(target_gg)} 抓取，得分: [{score_min:.3f}, {score_max:.3f}]")
            else:
                grasp_ids = target_grasp_ids
                if len(target_gg) > 0:
                    prev_target_count = len(target_gg)
                    prev_avg_score = target_gg.scores.mean()
                
                if frame_count % 15 == 1:
                    score_min, score_max = (target_gg.scores.min(), target_gg.scores.max()) if len(target_gg) > 0 else (0, 0)
                    print(f"帧 {frame_count}: 检测 {len(curr_gg)} 抓取, 追踪 {len(target_gg)} 抓取，得分: [{score_min:.3f}, {score_max:.3f}]")
            
            if args.debug and vis is not None:
                trans_mat = np.array([[1, 0, 0, 0], 
                                      [0, -1, 0, 0], 
                                      [0, 0, -1, 0], 
                                      [0, 0, 0, 1]])
                
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(points)
                cloud.colors = o3d.utility.Vector3dVector(colors)
                cloud.transform(trans_mat)
                
                grippers = target_gg.to_open3d_geometry_list() if len(target_gg) > 0 else []
                for gripper in grippers:
                    gripper.transform(trans_mat)
                
                vis.clear_geometries()
                vis.add_geometry(cloud)
                for gripper in grippers:
                    vis.add_geometry(gripper)
                vis.poll_events()
                vis.update_renderer()

        
    except KeyboardInterrupt:
        print(f"\n\n中断: 总共 {frame_count} 帧")
    finally:
        adapter.stop()
        if vis is not None:
            vis.destroy_window()


if __name__ == "__main__":
    run_tracking()
