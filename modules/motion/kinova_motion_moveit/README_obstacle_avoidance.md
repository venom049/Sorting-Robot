# 双机械臂自主避障位姿控制程序

## 概述

本程序实现了基于改进RRT*算法的双机械臂自主避障控制系统，能够在复杂环境中安全地将机械臂运动到目标位姿。

## 主要特性

- **改进的RRT*算法**: 使用优化的Rapidly-exploring Random Tree Star算法进行路径规划
- **动态避障**: 实时检测和规避环境中的障碍物
- **双臂协同**: 支持单臂和双臂协同避障运动
- **路径优化**: 自动寻找最优路径，最小化运动成本
- **安全机制**: 内置紧急停止和碰撞检测功能
- **实时重规划**: 支持动态环境下的路径重新规划

## 算法原理

### RRT*算法改进

1. **目标偏向采样**: 提高向目标区域探索的效率
2. **动态重布线半径**: 根据节点密度自适应调整重布线范围
3. **路径平滑**: 自动优化路径，减少不必要的转折
4. **多起点规划**: 支持从当前位置直接规划到目标

### 避障策略

- 环境建模为球形障碍物集合
- 路径段碰撞检测，确保路径安全性
- 目标点安全检查，自动寻找最近可达点
- 实时碰撞监控，支持动态障碍物

## 文件结构

```
motion_obstacle_avoidance.py    # 主程序文件
run_obstacle_avoidance.py       # 简化运行脚本
README_obstacle_avoidance.md     # 说明文档
```

## 依赖要求

```bash
# Python依赖
pip install numpy
```

ROS2依赖包：
- rclpy
- pymoveit2
- geometry_msgs
- sensor_msgs
- std_msgs
- visualization_msgs

## 实机部署步骤

### 1. 环境配置

```bash
# 运行环境配置脚本
cd /home/rpp/rpp_ws/src/robot/kinova_motion_moveit/kinova_motion_moveit/
./setup_environment.sh
```

### 2. 启动机械臂系统

```bash
# 终端1: 启动ROS2核心
source /opt/ros/humble/setup.bash
ros2 core

# 终端2: 启动机械臂驱动和MoveIt规划
cd /home/rpp/rpp_ws
source install/setup.bash
ros2 launch kinova_motion_moveit dual_arm_launch.py

# 终端3: 检查系统状态
ros2 node list
ros2 topic list | grep joint
```

### 3. 测试基本功能

```bash
# 进入程序目录
cd /home/rpp/rpp_ws/src/robot/kinova_motion_moveit/kinova_motion_moveit/

# 设置环境变量
source /home/rpp/rpp_ws/install/setup.bash

# 先运行简化测试版本
python3 test_obstacle_avoidance.py
```

### 4. 运行完整避障程序

```bash
# 基本运行
python3 motion_obstacle_avoidance.py

# 或使用简化脚本
python3 run_obstacle_avoidance.py --demo
```

## 使用方法

### 1. 基本运行

```bash
# 进入程序目录
cd /home/rpp/rpp_ws/src/robot/kinova_motion_moveit/kinova_motion_moveit/

# 设置环境变量
source /home/rpp/rpp_ws/install/setup.bash

# 运行演示场景
python3 motion_obstacle_avoidance.py
```

### 2. 使用简化脚本

```bash
# 运行预设演示场景
python3 run_obstacle_avoidance.py --demo

# 控制左臂到指定位姿 (x y z roll pitch yaw)
python3 run_obstacle_avoidance.py --left 0.4 0.3 0.6 0 0.785 0

# 控制右臂到指定位姿
python3 run_obstacle_avoidance.py --right -0.35 -0.25 0.5 0 -0.524 -0.785

# 双臂协同控制
python3 run_obstacle_avoidance.py --both 0.3 0.2 0.5 0 0.5 0.5 -0.3 -0.2 0.5 0 -0.5 -0.5
```

### 3. 使用ros2 run命令（推荐）

```bash
# 设置环境变量
source /home/rpp/rpp_ws/install/setup.bash

# 方法1: 使用快速启动脚本
cd /home/rpp/rpp_ws/src/robot/kinova_motion_moveit/kinova_motion_moveit/
./run_with_ros2.sh test    # 运行测试版本
./run_with_ros2.sh full     # 运行完整版本

# 方法2: 直接使用ros2 run命令
ros2 run kinova_motion_moveit test_obstacle_avoidance      # 简化测试版本
ros2 run kinova_motion_moveit motion_obstacle_avoidance    # 完整RRT*版本
```

### 4. 测试版本（推荐首次使用）

```bash
# 运行简化测试版本，使用MoveIt内置避障
python3 test_obstacle_avoidance.py
```

### 3. 参数说明

- **位置参数** (米): x, y, z
- **姿态参数** (弧度): roll, pitch, yaw
- **工作空间限制**: 
  - X, Y: ±0.895m
  - Z: -0.15m 到 1.1m

## 配置说明

### 机械臂参数

在 `DualArmObstacleAvoidanceController` 类中可以调整：

```python
# 运动参数
"max_velocity": 0.03,        # 最大速度 (m/s)
"max_acceleration": 0.01,    # 最大加速度 (m/s²)

# RRT*算法参数
max_iterations=3000,         # 最大迭代次数
step_size=0.03,              # 步长 (m)
goal_bias=0.15,              # 目标偏向概率
rewire_radius=0.15,          # 重布线半径 (m)
```

### 障碍物配置

在 `setup_environment_obstacles()` 方法中定义环境障碍物：

```python
obstacles = [
    {'center': [x, y, z], 'radius': r},  # 球形障碍物
    # 添加更多障碍物...
]
```

## 安全注意事项

1. **紧急停止**: 按 Ctrl+C 可立即停止所有运动
2. **速度限制**: 程序使用较低的安全速度
3. **碰撞检测**: 内置多重安全检查机制
4. **工作空间**: 自动检查目标点是否在可达范围内

## 故障排除

### 常见问题

1. **路径规划失败**
   - 检查目标点是否在工作空间内
   - 确认障碍物配置是否合理
   - 尝试调整RRT*参数

2. **运动执行失败**
   - 检查机械臂连接状态
   - 确认MoveIt2服务正常运行
   - 查看错误日志信息

3. **初始化失败**
   - 确认ROS2环境已正确配置
   - 检查机械臂驱动是否启动
   - 验证关节状态话题是否可用

### 调试模式

在代码中添加更多调试信息：

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 性能优化

1. **算法参数调优**:
   - 增加 `max_iterations` 提高成功率
   - 调整 `step_size` 平衡精度和速度
   - 优化 `goal_bias` 提高收敛速度

2. **并行计算**:
   - 双臂路径规划并行执行
   - 多线程碰撞检测

3. **内存管理**:
   - 定期清理RRT*节点树
   - 限制最大节点数量

## 扩展功能

### 支持的扩展

1. **动态障碍物**: 实时更新障碍物位置
2. **多臂协作**: 扩展到更多机械臂
3. **视觉集成**: 结合视觉系统进行障碍物检测
4. **学习算法**: 集成强化学习优化路径规划

### 自定义算法

可以在 `RRTStarPlanner` 类中实现其他算法：

```python
class CustomPlanner(RRTStarPlanner):
    def plan(self, start, goal):
        # 实现自定义算法
        pass
```

## 技术支持

如遇到问题，请检查：
1. ROS2环境配置
2. 机械臂硬件连接
3. MoveIt2配置文件
4. 程序日志输出

## 版本历史

- v1.0: 初始版本，实现基本RRT*避障算法
- 支持双臂协同控制
- 内置安全机制和紧急停止功能