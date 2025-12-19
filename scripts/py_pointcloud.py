#!/usr/bin/env python3
"""
获取相机内参，输出对齐的深度图和彩色图，生成点云
使用 pyrealsense2.pointcloud 生成点云并用 OpenCV 渲染
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import math

# 参考opencv_pointcloud_viewer.py处理点云渲染
class PointCloudState:
    def __init__(self, w, h):
        self.yaw = math.radians(-15)
        self.pitch = math.radians(-10)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.w = w
        self.h = h

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

def view(v, pc_state):
    return np.dot(v - pc_state.pivot, pc_state.rotation) + pc_state.pivot - pc_state.translation

def project(v, pc_state):
    h, w = pc_state.h, pc_state.w
    view_aspect = float(h)/w
    
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * (w*view_aspect, h) + (w/2.0, h/2.0)
    
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj

def render_pointcloud(out, verts, texcoords, color, pc_state):
    v = view(verts, pc_state)
    s = v[:, 2].argsort()[::-1]
    proj = project(v[s], pc_state)
    
    h, w = out.shape[:2]
    j, i = proj.astype(np.uint32).T
    
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm
    
    cw, ch = color.shape[:2][::-1]
    
    v_tex, u_tex = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    np.clip(u_tex, 0, ch-1, out=u_tex)
    np.clip(v_tex, 0, cw-1, out=v_tex)
    
    out[i[m], j[m]] = color[u_tex[m], v_tex[m]]


def main():
    parser = argparse.ArgumentParser(description='RealSense depth alignment and point cloud generation')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    parser.add_argument('--clip-distance', type=float, default=1.5, help='Background clipping distance in meters')
    args = parser.parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)  # 深度对齐到彩色
    
    pc = rs.pointcloud()
    
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"深度比例: {depth_scale}")
    print(f"背景裁剪距离: {args.clip_distance}米\n")
    
    frame_count = 0
    pc_state = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            frame_count += 1

            # 获得深度图和彩色图
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 通过 realsense2 库函数计算 points
            pc.map_to(color_frame)
            points = pc.calculate(aligned_depth_frame)
            
            # points.get_vertices() 和 points.get_texture_coordinates() 得到点和纹理坐标
            verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            texcoords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
            
            # 深度滤波
            mask_depth = verts[:, 2] < args.clip_distance
            verts_filtered = verts[mask_depth]
            texcoords_filtered = texcoords[mask_depth]
            
            if frame_count % 30 == 1:
                print(f"帧 {frame_count}: {verts.shape} -> 滤波后 {verts_filtered.shape}")
            
            # 可视化：将三维点用 opencv 二维渲染出来
            if not args.no_vis:
                h, w = color_image.shape[:2]
                
                if pc_state is None:
                    pc_state = PointCloudState(w, h)
                
                # 深度灰度图 | 点云渲染
                depth_colormap = cv2.cvtColor(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLOR_GRAY2BGR)
                pointcloud_render = np.zeros((h, w, 3), dtype=np.uint8)
                render_pointcloud(pointcloud_render, verts, texcoords, color_image, pc_state)
                
                cv2.imshow('RealSense: Depth | PointCloud', np.hstack((depth_colormap, pointcloud_render)))
        
            else:
                if frame_count >= 100:
                    print(f"\n已采集 {frame_count} 帧，退出...")
                    break

    finally:
        pipeline.stop()
        if not args.no_vis:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
