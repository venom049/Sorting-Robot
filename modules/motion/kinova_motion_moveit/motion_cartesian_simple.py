#!/usr/bin/env python3
"""
双机械臂笛卡尔空间位姿控制程序
直接输入笛卡尔空间的位姿坐标(x,y,z,roll,pitch,yaw)控制机械臂运动
使用欧拉角输入，自动转换为四元数
"""

import time
import threading
from threading import Thread
import math

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from pymoveit2 import MoveIt2
from geometry_msgs.msg import Pose


class DualArmCartesianPoseController:
    def __init__(self):
        rclpy.init()
        
        # 创建节点
        self.node = Node("dual_arm_cartesian_pose_controller")
        
        # 创建回调组
        self.callback_group = ReentrantCallbackGroup()
        
        # 机械臂配置
        self.arm_configs = {
            "left_arm": {
                "group_name": "left_arm",
                "base_link_name": "left_base_link",
                "end_effector_name": "left_tool_frame",
                "joint_names": [
                    "left_joint_1", "left_joint_2", "left_joint_3", 
                    "left_joint_4", "left_joint_5", "left_joint_6", "left_joint_7"
                ],
                "planner_id": "RRTConnect",
                "max_velocity": 0.05,
                "max_acceleration": 0.02
            },
            "right_arm": {
                "group_name": "right_arm",
                "base_link_name": "right_base_link",
                "end_effector_name": "right_tool_frame",
                "joint_names": [
                    "right_joint_1", "right_joint_2", "right_joint_3",
                    "right_joint_4", "right_joint_5", "right_joint_6", "right_joint_7"
                ],
                "planner_id": "RRTConnect",
                "max_velocity": 0.05,
                "max_acceleration": 0.02
            }
        }
        
        # 初始化机械臂控制器
        self.moveit2_left = MoveIt2(
            node=self.node,
            joint_names=self.arm_configs["left_arm"]["joint_names"],
            base_link_name=self.arm_configs["left_arm"]["base_link_name"],
            end_effector_name=self.arm_configs["left_arm"]["end_effector_name"],
            group_name=self.arm_configs["left_arm"]["group_name"],
            callback_group=self.callback_group
        )
        
        self.moveit2_right = MoveIt2(
            node=self.node,
            joint_names=self.arm_configs["right_arm"]["joint_names"],
            base_link_name=self.arm_configs["right_arm"]["base_link_name"],
            end_effector_name=self.arm_configs["right_arm"]["end_effector_name"],
            group_name=self.arm_configs["right_arm"]["group_name"],
            callback_group=self.callback_group
        )
        
        # 设置规划器参数
        self.moveit2_left.planner_id = self.arm_configs["left_arm"]["planner_id"]
        self.moveit2_left.max_velocity = self.arm_configs["left_arm"]["max_velocity"]
        self.moveit2_left.max_acceleration = self.arm_configs["left_arm"]["max_acceleration"]
        
        self.moveit2_right.planner_id = self.arm_configs["right_arm"]["planner_id"]
        self.moveit2_right.max_velocity = self.arm_configs["right_arm"]["max_velocity"]
        self.moveit2_right.max_acceleration = self.arm_configs["right_arm"]["max_acceleration"]
        
        # 启动执行器
        self.executor = MultiThreadedExecutor(2)
        self.executor.add_node(self.node)
        self.executor_thread = Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        
        # 等待初始化
        print("等待MoveIt2初始化...")
        for i in range(5):  # 尝试5次，每次等待1秒
            try:
                self.node.create_rate(1.0).sleep()
                print(f"初始化进度: {i+1}/5")
            except Exception as e:
                print(f"初始化等待出错: {e}")
                break
        
        print("双机械臂笛卡尔空间位姿控制器初始化完成")
        
        # 检查MoveIt2服务是否可用
        print("检查MoveIt2服务状态...")
        try:
            # 等待一段时间让MoveIt2服务完全启动
            time.sleep(2.0)
            print("MoveIt2服务检查完成")
        except Exception as e:
            print(f"MoveIt2服务检查出错: {e}")
    
    def wait_for_joint_states(self, timeout=10.0):
        """等待关节状态数据可用"""
        start_time = time.time()
        print(f"等待关节状态数据，超时时间: {timeout}秒...")
        
        while time.time() - start_time < timeout:
            left_state = self.moveit2_left.joint_state is not None
            right_state = self.moveit2_right.joint_state is not None
            
            if left_state and right_state:
                print("关节状态数据已就绪")
                return True
            
            # 打印等待状态
            elapsed = time.time() - start_time
            print(f"等待中... 已等待: {elapsed:.1f}秒, 左臂状态: {left_state}, 右臂状态: {right_state}")
            time.sleep(1.0)  # 增加等待间隔，减少打印频率
        
        print(f"等待关节状态超时，左臂状态: {self.moveit2_left.joint_state is not None}, 右臂状态: {self.moveit2_right.joint_state is not None}")
        return False
    
    def get_current_joint_positions(self, arm_name):
        """获取当前关节位置（弧度）"""
        if arm_name == "left_arm":
            moveit2 = self.moveit2_left
            joint_names = self.arm_configs["left_arm"]["joint_names"]
        else:
            moveit2 = self.moveit2_right
            joint_names = self.arm_configs["right_arm"]["joint_names"]
        
        if moveit2.joint_state is None:
            print(f"{arm_name} 关节状态不可用")
            return None
        
        positions = []
        for joint_name in joint_names:
            if joint_name in moveit2.joint_state.name:
                index = moveit2.joint_state.name.index(joint_name)
                positions.append(moveit2.joint_state.position[index])
            else:
                positions.append(0.0)
        
        return positions
    
    def print_joint_angles(self, arm_name, joint_positions, prefix=""):
        """打印关节角度信息"""
        print(f"{prefix}{arm_name}关节角度 (度):")
        for i, angle in enumerate(joint_positions):
            print(f"  关节{i+1}: {angle * 180.0 / 3.141592654:.2f}°")
    
    def check_workspace_limits(self, x, y, z):
        """
        检查位姿是否在EA200机械臂工作空间内
        
        工作空间限制（基于EA200机械臂规格）：
        - 最大伸展距离：895mm（从基座旋转中心到末端执行器）
        - 垂直工作范围：-150mm 到 +1100mm（相对于基座安装面）
        - 工作空间形状：近似不规则球体
        
        参数:
            x, y, z: 笛卡尔坐标（米）
        
        返回:
            bool: 是否在合理工作空间内
            str: 错误信息（如果超出限制）
        """
        # 计算到基座中心的水平距离（米）
        horizontal_distance = math.sqrt(x**2 + y**2)
        
        # 检查水平伸展距离（最大0.895米）
        if horizontal_distance > 0.895:
            return False, f"水平距离 {horizontal_distance:.3f}m 超出最大伸展距离 0.895m"
        
        # 检查垂直工作范围（-0.15m 到 1.1m）
        if z < -0.15:
            return False, f"Z坐标 {z:.3f}m 低于最低可达点 -0.15m"
        if z > 1.1:
            return False, f"Z坐标 {z:.3f}m 高于最高可达点 1.1m"
        
        # 检查基座附近不可达区域（半径0.1m内，Z<0.2m的区域）
        if horizontal_distance < 0.1 and z < 0.2:
            return False, "位置过于接近基座，可能存在碰撞风险"
        
        return True, "位姿在合理工作空间内"
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        将欧拉角转换为四元数
        
        坐标系说明：
        - 使用固定坐标系（世界坐标系）
        - 旋转顺序：ZYX（先绕Z轴yaw，再绕Y轴pitch，最后绕X轴roll）
        - 角度单位：弧度
        
        参数:
            roll: 绕X轴旋转角度（弧度）
            pitch: 绕Y轴旋转角度（弧度）
            yaw: 绕Z轴旋转角度（弧度）
        
        返回:
            list: [x, y, z, w] 四元数
        """
        # 计算半角
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        # 计算四元数分量
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return [x, y, z, w]
    
    def print_pose(self, pose, prefix=""):
        """打印位姿信息"""
        if pose:
            print(f"{prefix}笛卡尔位姿:")
            print(f"  位置 (x, y, z): ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}) m")
            print(f"  姿态 (x, y, z, w): ({pose.orientation.x:.3f}, {pose.orientation.y:.3f}, {pose.orientation.z:.3f}, {pose.orientation.w:.3f})")
        else:
            print(f"{prefix}位姿信息不可用")
    
    def create_pose_from_euler(self, x, y, z, roll_deg, pitch_deg, yaw_deg):
        """
        从位置和欧拉角创建位姿对象
        
        参数:
            x, y, z: 位置坐标（米）
            roll_deg: 绕X轴旋转角度（度）
            pitch_deg: 绕Y轴旋转角度（度）
            yaw_deg: 绕Z轴旋转角度（度）
        
        返回:
            Pose: 位姿对象
        """
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        
        # 将角度转换为弧度
        roll_rad = math.radians(float(roll_deg))
        pitch_rad = math.radians(float(pitch_deg))
        yaw_rad = math.radians(float(yaw_deg))
        
        # 将欧拉角转换为四元数
        quaternion = self.euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]
        
        return pose
    
    def move_to_pose(self, arm_name, target_pose):
        """移动机械臂到指定位姿"""
        if arm_name == "left_arm":
            moveit2 = self.moveit2_left
        else:
            moveit2 = self.moveit2_right
        
        try:
            # 使用MoveIt2的位姿控制功能
            moveit2.move_to_pose(
                position=[target_pose.position.x, target_pose.position.y, target_pose.position.z],
                quat_xyzw=[target_pose.orientation.x, target_pose.orientation.y, 
                          target_pose.orientation.z, target_pose.orientation.w]
            )
            
            # 等待执行完成
            success = moveit2.wait_until_executed()
            
            if success:
                print(f"{arm_name} 成功移动到目标位姿")
                return True
            else:
                print(f"{arm_name} 移动到目标位姿失败")
                return False
        except Exception as e:
            print(f"执行位姿运动时出错: {e}")
            return False
    
    def move_joints_relative(self, arm_name, joint_offsets_deg):
        """相对移动机械臂关节（角度）"""
        if arm_name == "left_arm":
            moveit2 = self.moveit2_left
        else:
            moveit2 = self.moveit2_right
        
        # 获取当前关节位置（弧度）
        current_positions_rad = self.get_current_joint_positions(arm_name)
        if not current_positions_rad:
            print(f"{arm_name} 无法获取当前位置")
            return False
        
        # 将当前弧度转换为角度用于显示
        current_positions_deg = [p * 180.0 / 3.141592654 for p in current_positions_rad]
        
        # 计算目标位置（当前角度 + 偏移角度）
        target_positions_deg = []
        for i, offset_deg in enumerate(joint_offsets_deg):
            if offset_deg is not None:
                # 使用偏移角度
                target_positions_deg.append(current_positions_deg[i] + offset_deg)
            else:
                # 使用当前角度
                target_positions_deg.append(current_positions_deg[i])
        
        # 转换为弧度
        target_positions_rad = [deg * 3.141592654 / 180.0 for deg in target_positions_deg]
        
        print(f"{arm_name} 当前角度: {[f'{p:.1f}°' for p in current_positions_deg]}")
        print(f"{arm_name} 关节偏移: {[f'{p:.1f}°' for p in joint_offsets_deg]}")
        print(f"{arm_name} 目标角度: {[f'{p:.1f}°' for p in target_positions_deg]}")
        
        # 执行运动
        moveit2.move_to_configuration(target_positions_rad)
        return moveit2.wait_until_executed()
    
    def select_arm(self):
        """选择要控制的机械臂"""
        print("=== 机械臂笛卡尔空间位姿控制程序 ===")
        print("请选择要控制的机械臂:")
        print("1. 左臂")
        print("2. 右臂")
        
        while True:
            try:
                choice = input("请输入选择 (1 或 2): ").strip()
                if choice == "1":
                    return "left_arm", "左臂"
                elif choice == "2":
                    return "right_arm", "右臂"
                else:
                    print("无效选择，请输入 1 或 2")
            except KeyboardInterrupt:
                print("\n程序被用户中断")
                return None, None
    
    def input_pose(self):
        """输入目标位姿坐标（使用欧拉角）"""
        print("\n请输入目标位姿坐标:")
        print("坐标系说明：")
        print("  - 位置：相对于机械臂基座坐标系")
        print("  - 姿态：使用欧拉角（度），旋转顺序为ZYX")
        print("    • Roll: 绕X轴旋转角度")
        print("    • Pitch: 绕Y轴旋转角度") 
        print("    • Yaw: 绕Z轴旋转角度")
        
        while True:
            try:
                x = input("\n位置 X (米): ").strip()
                y = input("位置 Y (米): ").strip()
                z = input("位置 Z (米): ").strip()
                roll = input("Roll (绕X轴旋转角度，度): ").strip()
                pitch = input("Pitch (绕Y轴旋转角度，度): ").strip()
                yaw = input("Yaw (绕Z轴旋转角度，度): ").strip()
                
                # 验证输入
                if not all([x, y, z, roll, pitch, yaw]):
                    print("所有参数都必须输入")
                    continue
                
                # 转换为浮点数
                x, y, z = float(x), float(y), float(z)
                roll, pitch, yaw = float(roll), float(pitch), float(yaw)
                
                # 检查工作空间限制
                is_valid, message = self.check_workspace_limits(x, y, z)
                if not is_valid:
                    print(f"\n⚠️  警告: {message}")
                    print("建议调整位置参数后重新输入")
                    continue
                
                # 创建位姿对象
                pose = self.create_pose_from_euler(x, y, z, roll, pitch, yaw)
                
                # 显示转换后的四元数（供调试参考）
                print(f"\n欧拉角转换结果:")
                print(f"  Roll: {roll}°, Pitch: {pitch}°, Yaw: {yaw}°")
                print(f"  四元数: x={pose.orientation.x:.4f}, y={pose.orientation.y:.4f}, z={pose.orientation.z:.4f}, w={pose.orientation.w:.4f}")
                
                return pose
                
            except ValueError:
                print("输入格式错误，请输入有效的数字")
                continue
            except KeyboardInterrupt:
                print("\n输入被用户中断")
                return None
            except Exception as e:
                print(f"输入处理出错: {e}")
                continue
    
    def execute_cartesian_pose_demo(self):
        """执行笛卡尔空间位姿控制演示"""
        # 选择要控制的机械臂
        arm_name, arm_display_name = self.select_arm()
        if not arm_name:
            return False
        
        print(f"\n控制对象: {arm_display_name}")
        
        # 等待关节状态数据
        print("等待关节状态数据...")
        if not self.wait_for_joint_states():
            print("等待关节状态超时，程序退出")
            return False
        
        # 获取初始关节位置
        print("\n初始状态:")
        initial_joints = self.get_current_joint_positions(arm_name)
        self.print_joint_angles(arm_display_name, initial_joints, "初始")
        
        if not initial_joints:
            print("无法获取初始关节位置，程序退出")
            return False
        
        # 输入目标位姿
        target_pose = self.input_pose()
        if not target_pose:
            print("无法获取目标位姿，程序退出")
            return False
        
        # 打印目标位姿
        print("\n目标位姿:")
        self.print_pose(target_pose, "目标")
        
        # 停顿2秒
        print("\n停顿2秒...")
        time.sleep(2)
        
        # 执行位姿运动
        if not self.move_to_pose(arm_name, target_pose):
            print("位姿运动失败，程序退出")
            return False
        
        # 获取运动后的关节位置
        print("\n运动后状态:")
        final_joints = self.get_current_joint_positions(arm_name)
        self.print_joint_angles(arm_display_name, final_joints, "运动后")
        
        print(f"\n{arm_display_name}笛卡尔空间位姿控制程序执行完成！")
        return True
    
    def shutdown(self):
        """关闭控制器"""
        try:
            if rclpy.ok():
                rclpy.shutdown()
            if self.executor_thread.is_alive():
                self.executor_thread.join(timeout=1.0)
        except Exception as e:
            print(f"关闭控制器时出错: {e}")


def main():
    controller = None
    try:
        # 创建控制器
        controller = DualArmCartesianPoseController()
        
        # 执行笛卡尔空间位姿控制演示
        controller.execute_cartesian_pose_demo()
        
    except Exception as e:
        print(f"程序执行出错: {e}")
    finally:
        # 关闭控制器
        if controller is not None:
            controller.shutdown()


if __name__ == "__main__":
    main()