# 快速开始指南

## 🚀 使用ros2 run命令运行避障控制

现在您可以像运行其他程序一样使用`ros2 run`命令：

### 1. 环境设置
```bash
cd /home/rpp/rpp_ws
source install/setup.bash
```

### 2. 运行避障控制程序

#### 简化测试版本（推荐首次使用）
```bash
ros2 run kinova_motion_moveit test_obstacle_avoidance
```

#### 完整RRT*避障版本
```bash
ros2 run kinova_motion_moveit motion_obstacle_avoidance
```

### 3. 使用快速启动脚本
```bash
cd /home/rpp/rpp_ws/src/robot/kinova_motion_moveit/kinova_motion_moveit/

# 运行测试版本
./run_with_ros2.sh test

# 运行完整版本
./run_with_ros2.sh full

# 查看帮助
./run_with_ros2.sh help
```

## 📋 可用的所有命令

```bash
# 原有的程序
ros2 run kinova_motion_moveit motion_joint_dualarm
ros2 run kinova_motion_moveit motion_cartesian_pose
ros2 run kinova_motion_moveit motion_cartesian_roundtrip
ros2 run kinova_motion_moveit motion_trajectory_dualarm
ros2 run kinova_motion_moveit motion_joint_dualarm_hand

# 新增的避障程序
ros2 run kinova_motion_moveit test_obstacle_avoidance
ros2 run kinova_motion_moveit motion_obstacle_avoidance
```

## ⚠️ 运行前检查

确保以下条件已满足：
1. 机械臂硬件已连接
2. MoveIt规划节点正在运行
3. 关节状态数据正常发布

## 🛑 紧急停止

在任何程序运行时，按 `Ctrl+C` 可立即停止机械臂运动。