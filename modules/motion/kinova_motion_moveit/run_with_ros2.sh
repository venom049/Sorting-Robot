#!/bin/bash

# 使用ros2 run命令启动避障控制程序的脚本

echo "=== 双机械臂避障控制ROS2启动脚本 ==="

# 检查ROS2环境
if [ -z "$ROS_DISTRO" ]; then
    echo "错误: ROS2环境未配置"
    echo "请先运行: source /opt/ros/humble/setup.bash"
    exit 1
fi

# 设置工作空间环境
cd /home/rpp/rpp_ws
source install/setup.bash

echo "当前ROS2版本: $ROS_DISTRO"
echo "工作空间: $(pwd)"
echo ""

# 显示可用命令
echo "可用的避障控制命令："
echo "1. ros2 run kinova_motion_moveit test_obstacle_avoidance     # 简化测试版本"
echo "2. ros2 run kinova_motion_moveit motion_obstacle_avoidance   # 完整RRT*版本"
echo ""

# 根据参数选择运行哪个程序
case "$1" in
    "test"|"simple"|"1")
        echo "启动简化测试版本..."
        ros2 run kinova_motion_moveit test_obstacle_avoidance
        ;;
    "full"|"rrt"|"2"|"")
        echo "启动完整RRT*避障版本..."
        ros2 run kinova_motion_moveit motion_obstacle_avoidance
        ;;
    "help"|"-h"|"--help")
        echo "使用方法:"
        echo "  $0 [选项]"
        echo ""
        echo "选项:"
        echo "  test, simple, 1    - 运行简化测试版本"
        echo "  full, rrt, 2       - 运行完整RRT*版本 (默认)"
        echo "  help, -h, --help    - 显示此帮助信息"
        echo ""
        echo "示例:"
        echo "  $0 test     # 运行测试版本"
        echo "  $0 full     # 运行完整版本"
        echo "  $0          # 运行完整版本 (默认)"
        ;;
    *)
        echo "错误: 未知选项 '$1'"
        echo "使用 '$0 help' 查看帮助信息"
        exit 1
        ;;
esac