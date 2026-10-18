@echo off
REM Windows下运行目标位姿控制程序的启动脚本

echo === 目标位姿控制程序启动脚本 ===

REM 检查ROS2环境
if "%ROS_DISTRO%"=="" (
    echo 错误: ROS2环境未配置
    echo 请先运行: call C:\opt\ros\humble\setup.bat
    pause
    exit /b 1
)

echo 当前ROS2版本: %ROS_DISTRO%
echo 当前目录: %CD%
echo.

REM 显示使用说明
echo 目标位姿控制程序使用说明：
echo.
echo 运行前请确保：
echo 1. ROS2环境已配置 ✓
echo 2. 机械臂硬件已连接并启动
echo 3. MoveIt规划节点正在运行
echo 4. 关节状态数据正常发布
echo.
echo 检查当前ROS2状态...
echo.

REM 检查关键ROS2节点和话题
echo 检查ROS2节点：
ros2 node list 2>nul
if %errorlevel% neq 0 (
    echo ⚠ 无法获取ROS2节点列表，请检查ROS2是否正在运行
    echo 建议运行以下命令启动ROS2：
    echo   call C:\opt\ros\humble\setup.bat
    echo   ros2 daemon start
    pause
    exit /b 1
)

echo.
echo 检查机械臂相关话题：
ros2 topic list | findstr joint_states
if %errorlevel% neq 0 (
    echo ⚠ joint_states话题未找到
    echo 请确保机械臂驱动节点正在运行
) else (
    echo ✓ joint_states话题可用
)

ros2 topic list | findstr tf
if %errorlevel% neq 0 (
    echo ⚠ tf话题未找到
) else (
    echo ✓ tf话题可用
)

echo.
echo === 启动目标位姿控制程序 ===
echo.

REM 运行目标位姿控制程序
python motion_target_pose.py

if %errorlevel% neq 0 (
    echo.
    echo 程序运行出错，可能的解决方案：
    echo 1. 检查Python环境：python --version
    echo 2. 检查ROS2环境：ros2 --version
    echo 3. 检查依赖包：pip list ^| findstr pymoveit2
    echo 4. 确保机械臂节点正在运行
    echo 5. 运行环境配置脚本：setup_windows_environment.bat
    pause
)

pause