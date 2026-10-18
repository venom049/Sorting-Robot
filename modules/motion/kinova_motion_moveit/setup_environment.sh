#!/bin/bash

# 双机械臂避障控制环境配置脚本
# 用于在实机上配置运行环境

echo "=== 双机械臂避障控制环境配置 ==="

# 检查ROS2环境
if [ -z "$ROS_DISTRO" ]; then
    echo "错误: ROS2环境未配置"
    echo "请先运行: source /opt/ros/humble/setup.bash"
    exit 1
fi

echo "当前ROS2版本: $ROS_DISTRO"

# 进入工作空间
cd /home/rpp/rpp_ws

# 检查工作空间
if [ ! -f "setup.py" ]; then
    echo "错误: 未找到工作空间setup.py文件"
    exit 1
fi

echo "检查工作空间依赖..."

# 安装Python依赖
echo "安装Python依赖..."
pip3 install numpy 2>/dev/null || {
    echo "警告: numpy安装失败，请手动安装"
}

# 检查必要的ROS2包
echo "检查ROS2包..."
ros2 pkg list | grep -q "pymoveit2" || {
    echo "警告: pymoveit2包未找到"
    echo "请确保已安装pymoveit2: sudo apt install ros-$ROS_DISTRO-pymoveit2"
}

ros2 pkg list | grep -q "geometry_msgs" || {
    echo "警告: geometry_msgs包未找到"
}

ros2 pkg list | grep -q "sensor_msgs" || {
    echo "警告: sensor_msgs包未找到"
}

# 编译工作空间
echo "编译工作空间..."
colcon build --packages-select kinova_motion_moveit 2>/dev/null || {
    echo "编译失败，尝试完整编译..."
    colcon build
}

# 设置环境
echo "设置环境变量..."
source install/setup.bash

# 检查机械臂节点
echo "检查机械臂节点状态..."
ros2 node list | grep -q "move_group" && {
    echo "✓ MoveIt move_group节点正在运行"
} || {
    echo "⚠ MoveIt move_group节点未运行"
    echo "请先启动机械臂控制节点"
}

# 检查关节状态
echo "检查关节状态话题..."
ros2 topic list | grep -q "joint_states" && {
    echo "✓ joint_states话题可用"
} || {
    echo "⚠ joint_states话题未找到"
}

# 检查TF变换
echo "检查TF变换..."
ros2 topic list | grep -q "tf" && {
    echo "✓ tf话题可用"
} || {
    echo "⚠ tf话题未找到"
}

echo ""
echo "=== 环境配置完成 ==="
echo ""
echo "运行程序前请确保："
echo "1. 机械臂硬件已连接并启动"
echo "2. MoveIt规划节点正在运行"
echo "3. 关节状态数据正常发布"
echo ""
echo "运行命令："
echo "cd /home/rpp/rpp_ws/src/robot/kinova_motion_moveit/kinova_motion_moveit/"
echo "source /home/rpp/rpp_ws/install/setup.bash"
echo "python3 motion_obstacle_avoidance.py"