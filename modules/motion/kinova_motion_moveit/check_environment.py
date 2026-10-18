#!/usr/bin/env python3
"""
环境检查脚本
用于诊断ROS2环境和机械臂连接状态
"""

import subprocess
import sys
import os


def run_command(cmd, timeout=5):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "命令超时"
    except Exception as e:
        return False, "", str(e)


def check_ros2_environment():
    """检查ROS2环境"""
    print("=== ROS2环境检查 ===")
    
    # 检查ROS_DISTRO
    ros_distro = os.environ.get('ROS_DISTRO')
    if ros_distro:
        print(f"✓ ROS2版本: {ros_distro}")
    else:
        print("✗ ROS2环境未配置")
        print("  解决方案: 运行 'call C:\\opt\\ros\\humble\\setup.bat'")
        return False
    
    # 检查ros2命令
    success, stdout, stderr = run_command("ros2 --version")
    if success:
        print(f"✓ ros2命令可用: {stdout}")
    else:
        print("✗ ros2命令不可用")
        print("  解决方案: 确保ROS2正确安装并配置PATH")
        return False
    
    return True


def check_ros2_daemon():
    """检查ROS2守护进程"""
    print("\n=== ROS2守护进程检查 ===")
    
    # 尝试列出节点
    success, stdout, stderr = run_command("ros2 node list")
    if success:
        print("✓ ROS2守护进程正在运行")
        nodes = stdout.split('\n') if stdout else []
        print(f"  当前节点数: {len(nodes)}")
        if nodes:
            print("  活跃节点:")
            for node in nodes[:5]:  # 只显示前5个
                print(f"    - {node}")
            if len(nodes) > 5:
                print(f"    ... 还有 {len(nodes)-5} 个节点")
    else:
        print("✗ ROS2守护进程未运行")
        print("  解决方案: 运行 'ros2 daemon start'")
        return False
    
    return True


def check_topics():
    """检查关键话题"""
    print("\n=== 关键话题检查 ===")
    
    # 检查joint_states话题
    success, stdout, stderr = run_command("ros2 topic list | findstr joint_states")
    if success and stdout:
        print("✓ joint_states话题可用")
        # 尝试获取话题信息
        success2, stdout2, stderr2 = run_command("ros2 topic info /joint_states")
        if success2:
            print(f"  话题信息: {stdout2}")
    else:
        print("✗ joint_states话题未找到")
        print("  解决方案: 启动机器人驱动节点")
    
    # 检查tf话题
    success, stdout, stderr = run_command("ros2 topic list | findstr tf")
    if success and stdout:
        print("✓ tf话题可用")
    else:
        print("✗ tf话题未找到")
        print("  解决方案: 启动机器人状态发布器")
    
    # 检查move_group服务
    success, stdout, stderr = run_command("ros2 service list | findstr move_group")
    if success and stdout:
        print("✓ MoveIt服务可用")
    else:
        print("✗ MoveIt服务未找到")
        print("  解决方案: 启动MoveIt规划节点")


def check_python_packages():
    """检查Python包"""
    print("\n=== Python包检查 ===")
    
    required_packages = [
        ('rclpy', 'ROS2 Python客户端'),
        ('numpy', '数值计算库'),
        ('geometry_msgs', '几何消息类型'),
    ]
    
    optional_packages = [
        ('pymoveit2', 'MoveIt Python接口'),
    ]
    
    print("必需包:")
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}: {description}")
        except ImportError:
            print(f"✗ {package}: {description}")
            print(f"  安装: pip install {package}")
    
    print("\n可选包:")
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"✓ {package}: {description}")
        except ImportError:
            print(f"✗ {package}: {description}")
            print(f"  安装: pip install {package}")


def check_moveit_status():
    """检查MoveIt状态"""
    print("\n=== MoveIt状态检查 ===")
    
    # 检查move_group节点
    success, stdout, stderr = run_command("ros2 node list | findstr move_group")
    if success and stdout:
        print("✓ move_group节点正在运行")
        
        # 检查MoveIt动作服务
        success, stdout, stderr = run_command("ros2 action list | findstr move_group")
        if success and stdout:
            print("✓ MoveIt动作服务可用")
            actions = stdout.split('\n') if stdout else []
            for action in actions[:3]:  # 显示前3个
                print(f"    - {action}")
        else:
            print("✗ MoveIt动作服务未找到")
    else:
        print("✗ move_group节点未运行")
        print("  解决方案: 启动MoveIt规划节点")


def suggest_solutions():
    """提供解决方案建议"""
    print("\n=== 解决方案建议 ===")
    
    print("如果检查失败，请按以下步骤操作：")
    print()
    print("1. 配置ROS2环境:")
    print("   call C:\\opt\\ros\\humble\\setup.bat")
    print()
    print("2. 启动ROS2守护进程:")
    print("   ros2 daemon start")
    print()
    print("3. 安装Python依赖:")
    print("   pip install numpy pymoveit2")
    print()
    print("4. 启动机器人节点 (需要根据您的机器人配置调整):")
    print("   # 示例命令")
    print("   ros2 launch your_robot_robot_moveit_config move_group.launch.py")
    print()
    print("5. 运行环境检查:")
    print("   python check_environment.py")
    print()
    print("6. 运行测试程序:")
    print("   python test_target_pose.py")
    print()
    print("7. 运行主程序:")
    print("   python motion_target_pose.py")


def main():
    """主检查流程"""
    print("ROS2环境和机械臂连接状态检查")
    print("=" * 50)
    
    # 执行各项检查
    ros2_ok = check_ros2_environment()
    if not ros2_ok:
        print("\n基础环境检查失败，请先解决ROS2环境问题")
        suggest_solutions()
        return
    
    daemon_ok = check_ros2_daemon()
    if not daemon_ok:
        print("\nROS2守护进程检查失败")
        suggest_solutions()
        return
    
    check_topics()
    check_python_packages()
    check_moveit_status()
    
    print("\n" + "=" * 50)
    print("环境检查完成")
    suggest_solutions()
    
    # 提供下一步操作建议
    print("\n下一步操作:")
    print("1. 如果所有检查都通过: 运行 'python motion_target_pose.py'")
    print("2. 如果部分检查失败: 按照上述解决方案修复")
    print("3. 如果ROS2相关问题: 运行 'python test_target_pose.py' 进行逻辑测试")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n检查被用户中断")
    except Exception as e:
        print(f"\n检查过程中出现错误: {e}")
        print("请检查Python环境和权限设置")