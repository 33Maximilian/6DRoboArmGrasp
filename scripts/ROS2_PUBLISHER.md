# ROS2 话题验证指南

## 快速验证流程

### 1. 启动tracking.py发布话题

```bash
cd /home/gpu00-905/6DRoboArmGrasp
python3 scripts/tracking.py --checkpoint-path anygrasp_sdk/grasp_tracking/log/checkpoint_tracking.tar
```

### 2. 验证话题发布

```bash
# 列出所有话题
ros2 topic list | grep grasp

# 预期输出
/grasp/position
/grasp/rotation
/grasp/width
/grasp/score
```

### 3. 查看话题内容

```bash
# 订阅位置话题（会持续输出）
ros2 topic echo /grasp/position

# 订阅姿态话题
ros2 topic echo /grasp/rotation

# 订阅宽度话题
ros2 topic echo /grasp/width

# 订阅评分话题
ros2 topic echo /grasp/score
```

### 4. 查看话题信息

```bash
# 查看话题的消息类型和发布频率
ros2 topic info /grasp/position

## 消息类型说明

| 话题 | 消息类型 | 字段 |
|------|---------|------|
| `/grasp/position` | geometry_msgs/PointStamped | header, point (x, y, z) |
| `/grasp/rotation` | geometry_msgs/QuaternionStamped | header, quaternion (x, y, z, w) |
| `/grasp/width` | std_msgs/Float64 | data |
| `/grasp/score` | std_msgs/Float64 | data |
```

## 发布频率

- 话题发布频率：**30Hz** (每33ms一次)
- 每次发布4个话题（位置、姿态、宽度、评分）
- 质量监测：追踪丢失>50% 或评分下降>40% 时触发重检
