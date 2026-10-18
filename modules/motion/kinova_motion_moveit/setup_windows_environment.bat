@echo off
REM 双机械臂避障控制Windows环境配置脚本

echo === 双机械臂避障控制Windows环境配置 ===

REM 检查ROS2环境
if "%ROS_DISTRO%"=="" (
    echo 错误: ROS2环境未配置
    echo 请先运行: call C:\opt\ros\humble\setup.bat
    pause
    exit /b 1
)

echo 当前ROS2版本: %ROS_DISTRO%

REM 检查Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: Python未安装或不在PATH中
    pause
    exit /b 1
)

echo 检查Python依赖...

REM 检查必要的Python包
python -c "import numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo 警告: numpy未安装，正在安装...
    pip install numpy
)

python -c "import rclpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo 警告: rclpy未安装，请检查ROS2 Python安装
)

python -c "import pymoveit2" >nul 2>&1
if %errorlevel% neq 0 (
    echo 警告: pymoveit2未找到
    echo 请确保已安装: pip install pymoveit2
)

echo ""
echo === 环境检查完成 ===
echo ""
echo 运行程序前请确保：
echo 1. 机械臂硬件已连接并启动
echo 2. MoveIt规划节点正在运行
echo 3. 关节状态数据正常发布
echo ""
echo 检查ROS2节点状态...
ros2 node list 2>nul
if %errorlevel% equ 0 (
    echo ✓ ros2节点列表可以获取
) else (
    echo ⚠ 无法获取ROS2节点列表，请检查ROS2是否正常运行
)

echo 检查关节状态话题...
ros2 topic list | findstr joint_states >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ joint_states话题可用
) else (
    echo ⚠ joint_states话题未找到
    echo 请确保机械臂驱动节点正在运行
)

pause