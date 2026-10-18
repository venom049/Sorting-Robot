#!/usr/bin/env python3
"""
双机械臂笛卡尔空间位姿控制程序 (修复版2 - 修复IK解算数据映射错误)
解决问题：修复了IK服务返回全机器人关节数据时，因顺序不一致导致的关节归零问题
"""

import time
import threading
from threading import Thread
import math
try:
    import numpy as np
except ImportError:
    pass

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from pymoveit2 import MoveIt2
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState

class DualArmCartesianPoseController:
    def __init__(self):
        rclpy.init()
        
        self.node = Node("dual_arm_cartesian_pose_controller")
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
                "planner_id": "RRTstarkConfigDefault",
                "max_velocity": 0.2,
                "max_acceleration": 0.03
            },
            "right_arm": {
                "group_name": "right_arm",
                "base_link_name": "right_base_link",
                "end_effector_name": "right_tool_frame",
                "joint_names": [
                    "right_joint_1", "right_joint_2", "right_joint_3",
                    "right_joint_4", "right_joint_5", "right_joint_6", "right_joint_7"
                ],
                "planner_id": "RRTstarkConfigDefault",
                "max_velocity": 0.2,
                "max_acceleration": 0.03
            }
        }
        
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

        # 初始化逆解服务客户端
        self.ik_client = self.node.create_client(GetPositionIK, '/compute_ik', callback_group=self.callback_group)
        
        self.executor = MultiThreadedExecutor(2)
        self.executor.add_node(self.node)
        self.executor_thread = Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        
        print("等待MoveIt2初始化...")
        for i in range(5):  
            try:
                self.node.create_rate(1.0).sleep()
                print(f"初始化进度: {i+1}/5")
            except Exception as e:
                print(f"初始化等待出错: {e}")
                break
        
        self.init_safety_wall()

        print("双机械臂笛卡尔空间位姿控制器初始化完成")
        
        try:
            time.sleep(2.0)
            if not self.ik_client.wait_for_service(timeout_sec=2.0):
                 print("⚠️ 警告: /compute_ik 服务未就绪")
            else:
                 print("✅ 逆解计算服务 (IK Service) 连接成功")
            print("MoveIt2服务检查完成")
        except Exception as e:
            print(f"MoveIt2服务检查出错: {e}")
    
    def init_safety_wall(self):
        print("正在构建安全虚拟墙 (X > -0.35限制)...")
        try:
            wall_thickness = 1.0 
            wall_center_x = -0.35 - (wall_thickness / 2.0)
            wall_size = [wall_thickness, 4.0, 4.0]
            wall_pose = [wall_center_x, 0.0, 1.0] 
            frame_id = "left_base_link" 
            
            self.moveit2_left.add_collision_box(
                id="safety_limit_wall", 
                size=wall_size, 
                position=wall_pose,
                quat_xyzw=[0.0, 0.0, 0.0, 1.0], 
                frame_id=frame_id
            )
            time.sleep(0.5) 
            print("✅ 安全虚拟墙已添加，RRT* 算法将自动避开 X < -0.35 区域")
        except Exception as e:
            print(f"⚠️ 警告: 安全墙添加失败，请注意手动安全! 错误: {e}")

    def wait_for_joint_states(self, timeout=10.0):
        start_time = time.time()
        print(f"等待关节状态数据，超时时间: {timeout}秒...")
        while time.time() - start_time < timeout:
            left_state = self.moveit2_left.joint_state is not None
            right_state = self.moveit2_right.joint_state is not None
            if left_state and right_state:
                print("关节状态数据已就绪")
                return True
            time.sleep(1.0)
        return False
    
    def get_current_joint_positions(self, arm_name):
        if arm_name == "left_arm":
            moveit2 = self.moveit2_left
            joint_names = self.arm_configs["left_arm"]["joint_names"]
        else:
            moveit2 = self.moveit2_right
            joint_names = self.arm_configs["right_arm"]["joint_names"]
        
        if moveit2.joint_state is None:
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
        print(f"{prefix}{arm_name}关节角度 (度):")
        for i, angle in enumerate(joint_positions):
            print(f"  关节{i+1}: {angle * 180.0 / 3.141592654:.2f}°")
    
    def check_workspace_limits(self, x, y, z):
        if x <= -0.35:
            return False, f"违反安全限制: X坐标 {x:.3f}m 必须大于 -0.35m"
        horizontal_distance = math.sqrt(x**2 + y**2)
        if horizontal_distance > 0.895:
            return False, f"水平距离 {horizontal_distance:.3f}m 超出最大伸展距离 0.895m"
        if z < -0.15:
            return False, f"Z坐标 {z:.3f}m 低于最低可达点 -0.15m"
        if z > 1.1:
            return False, f"Z坐标 {z:.3f}m 高于最高可达点 1.1m"
        if horizontal_distance < 0.1 and z < 0.2:
            return False, "位置过于接近基座，可能存在碰撞风险"
        return True, "位姿在合理工作空间内"
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return [x, y, z, w]
    
    def print_pose(self, pose, prefix=""):
        if pose:
            print(f"{prefix}笛卡尔位姿:")
            print(f"  位置 (x, y, z): ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}) m")
            print(f"  姿态 (x, y, z, w): ({pose.orientation.x:.3f}, {pose.orientation.y:.3f}, {pose.orientation.z:.3f}, {pose.orientation.w:.3f})")
        else:
            print(f"{prefix}位姿信息不可用")
    
    def create_pose_from_euler(self, x, y, z, roll_deg, pitch_deg, yaw_deg):
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        roll_rad = math.radians(float(roll_deg))
        pitch_rad = math.radians(float(pitch_deg))
        yaw_rad = math.radians(float(yaw_deg))
        quaternion = self.euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]
        return pose
    
    def compute_ik_solution(self, arm_name, target_pose, seed_joints):
        """
        [算法核心修复] 逆运动学求解
        修复了返回数据映射错误的问题：必须通过关节名称匹配来提取正确的关节角度
        """
        req = PositionIKRequest()
        req.group_name = self.arm_configs[arm_name]["group_name"]
        req.robot_state = RobotState()
        
        # 填充 Seed State
        req.robot_state.joint_state.name = self.arm_configs[arm_name]["joint_names"]
        req.robot_state.joint_state.position = seed_joints
        
        # 设置目标位姿
        req.pose_stamped.header.frame_id = self.arm_configs[arm_name]["base_link_name"]
        req.pose_stamped.pose = target_pose
        req.timeout.sec = 0
        req.timeout.nanosec = 50000000 
        req.avoid_collisions = True 

        try:
            future = self.ik_client.call_async(GetPositionIK.Request(ik_request=req))
            while rclpy.ok():
                if future.done():
                    response = future.result()
                    if response.error_code.val == 1: 
                        # [关键修复] 解析返回的全机器人状态，提取当前手臂的关节
                        # response.solution.joint_state 包含所有关节（可能包含左右臂），顺序不确定
                        full_state_names = response.solution.joint_state.name
                        full_state_positions = response.solution.joint_state.position
                        
                        # 构建 名字->角度 的字典
                        state_dict = dict(zip(full_state_names, full_state_positions))
                        
                        # 按我们定义的关节顺序提取数据
                        target_joint_values = []
                        target_names = self.arm_configs[arm_name]["joint_names"]
                        
                        for name in target_names:
                            if name in state_dict:
                                target_joint_values.append(state_dict[name])
                            else:
                                print(f"⚠️ 警告: IK返回结果中缺失关节 {name}，默认为0")
                                target_joint_values.append(0.0)
                                
                        return target_joint_values
                    else:
                        print(f"IK求解失败，错误码: {response.error_code.val}")
                        return None
                time.sleep(0.01)
        except Exception as e:
            print(f"调用IK服务出错: {e}")
            return None

    def optimize_joint_continuity(self, current_joints, target_joints):
        """
        [算法核心] 关节空间连续性优化
        解决 179° -> -179° 问题
        """
        optimized_joints = []
        for curr, targ in zip(current_joints, target_joints):
            diff = targ - curr
            # 归一化到 [-pi, pi]
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            # 最优目标 = 当前 + 最小偏差
            optimized_val = curr + diff
            optimized_joints.append(optimized_val)
        return optimized_joints

    def move_to_pose(self, arm_name, target_pose):
        """
        [智能位姿运动控制]
        """
        if arm_name == "left_arm":
            moveit2 = self.moveit2_left
        else:
            moveit2 = self.moveit2_right
        
        print(f"[{arm_name}] 正在计算最优路径 (IK Seeding + Joint Optimization)...")
        
        # 1. 获取当前关节作为种子 (Seed)
        current_joints = self.get_current_joint_positions(arm_name)
        if not current_joints:
            print("无法获取当前关节，无法进行优化规划。")
            return False

        # 2. 计算最优逆解 (使用当前关节作为参考)
        ik_solution = self.compute_ik_solution(arm_name, target_pose, current_joints)
        
        if ik_solution:
            print(f"[{arm_name}] ✅ 逆解计算成功，正在优化关节轨迹...")
            
            # 3. 优化关节连续性
            final_joints = self.optimize_joint_continuity(current_joints, ik_solution)
            
            diffs = [math.degrees(f - c) for f, c in zip(final_joints, current_joints)]
            max_diff = max(map(abs, diffs))
            print(f"    最大关节变动量: {max_diff:.2f}°")
            
            # 4. 执行关节空间运动
            moveit2.move_to_configuration(final_joints)
            success = moveit2.wait_until_executed()
            
            if success:
                print(f"[{arm_name}] ✅ 运动执行成功")
                return True
            else:
                print(f"[{arm_name}] ❌ 移动失败")
                return False
        else:
            # IK服务失败时的保底策略
            print(f"[{arm_name}] ⚠️ IK求解失败，尝试原始规划器...")
            moveit2.move_to_pose(
                position=[target_pose.position.x, target_pose.position.y, target_pose.position.z],
                quat_xyzw=[target_pose.orientation.x, target_pose.orientation.y, 
                          target_pose.orientation.z, target_pose.orientation.w]
            )
            return moveit2.wait_until_executed()
    
    def move_joints_relative(self, arm_name, joint_offsets_deg):
        """相对移动机械臂关节（角度）"""
        if arm_name == "left_arm":
            moveit2 = self.moveit2_left
        else:
            moveit2 = self.moveit2_right
        
        current_positions_rad = self.get_current_joint_positions(arm_name)
        if not current_positions_rad:
            print(f"{arm_name} 无法获取当前位置")
            return False
        
        current_positions_deg = [p * 180.0 / 3.141592654 for p in current_positions_rad]
        
        target_positions_deg = []
        for i, offset_deg in enumerate(joint_offsets_deg):
            if offset_deg is not None:
                target_positions_deg.append(current_positions_deg[i] + offset_deg)
            else:
                target_positions_deg.append(current_positions_deg[i])
        
        target_positions_rad = [deg * 3.141592654 / 180.0 for deg in target_positions_deg]
        
        print(f"{arm_name} 当前角度: {[f'{p:.1f}°' for p in current_positions_deg]}")
        print(f"{arm_name} 关节偏移: {[f'{p:.1f}°' for p in joint_offsets_deg]}")
        print(f"{arm_name} 目标角度: {[f'{p:.1f}°' for p in target_positions_deg]}")
        
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
                
                if not all([x, y, z, roll, pitch, yaw]):
                    print("所有参数都必须输入")
                    continue
                
                x, y, z = float(x), float(y), float(z)
                roll, pitch, yaw = float(roll), float(pitch), float(yaw)
                
                is_valid, message = self.check_workspace_limits(x, y, z)
                if not is_valid:
                    print(f"\n⚠️  警告: {message}")
                    print("建议调整位置参数后重新输入")
                    continue
                
                pose = self.create_pose_from_euler(x, y, z, roll, pitch, yaw)
                
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
        arm_name, arm_display_name = self.select_arm()
        if not arm_name:
            return False
        
        print(f"\n控制对象: {arm_display_name}")
        print("等待关节状态数据...")
        if not self.wait_for_joint_states():
            print("等待关节状态超时，程序退出")
            return False
        
        print("\n初始状态:")
        initial_joints = self.get_current_joint_positions(arm_name)
        self.print_joint_angles(arm_display_name, initial_joints, "初始")
        
        if not initial_joints:
            print("无法获取初始关节位置，程序退出")
            return False
        
        target_pose = self.input_pose()
        if not target_pose:
            print("无法获取目标位姿，程序退出")
            return False
        
        print("\n目标位姿:")
        self.print_pose(target_pose, "目标")
        
        print("\n停顿2秒...")
        time.sleep(2)
        
        if not self.move_to_pose(arm_name, target_pose):
            print("位姿运动失败，程序退出")
            return False
        
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
        controller = DualArmCartesianPoseController()
        controller.execute_cartesian_pose_demo()
    except Exception as e:
        print(f"程序执行出错: {e}")
    finally:
        if controller is not None:
            controller.shutdown()


if __name__ == "__main__":
    main()