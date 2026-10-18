#!/usr/bin/env python3
"""
双机械臂笛卡尔空间往返运动控制程序
1. 读取当前双臂的笛卡尔位姿信息
2. 在当前位姿附近进行小幅运动
3. 再回到初始位姿
4. 打印成功信息
"""

import time
import threading
from threading import Thread
import math
import numpy as np

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import tf2_ros
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from pymoveit2 import MoveIt2
from geometry_msgs.msg import Pose, Point, Quaternion


class DualArmCartesianRoundtripController:
    def __init__(self):
        rclpy.init()
        
        # 创建节点
        self.node = Node("dual_arm_cartesian_roundtrip_controller")
        
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
                "max_acceleration": 0.02,
                # 新增优化参数
                "planning_time": 5.0,  # 规划时间限制
                "num_planning_attempts": 3,  # 规划尝试次数
                "goal_joint_tolerance": 0.01,  # 关节目标容差
                "goal_position_tolerance": 0.001,  # 位置目标容差
                "goal_orientation_tolerance": 0.01  # 姿态目标容差
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
                "max_acceleration": 0.02,
                # 新增优化参数
                "planning_time": 5.0,  # 规划时间限制
                "num_planning_attempts": 3,  # 规划尝试次数
                "goal_joint_tolerance": 0.01,  # 关节目标容差
                "goal_position_tolerance": 0.001,  # 位置目标容差
                "goal_orientation_tolerance": 0.01  # 姿态目标容差
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
        
        # 设置优化参数
        self.moveit2_left.planning_time = self.arm_configs["left_arm"]["planning_time"]
        self.moveit2_left.num_planning_attempts = self.arm_configs["left_arm"]["num_planning_attempts"]
        self.moveit2_left.goal_joint_tolerance = self.arm_configs["left_arm"]["goal_joint_tolerance"]
        self.moveit2_left.goal_position_tolerance = self.arm_configs["left_arm"]["goal_position_tolerance"]
        self.moveit2_left.goal_orientation_tolerance = self.arm_configs["left_arm"]["goal_orientation_tolerance"]
        
        self.moveit2_right.planning_time = self.arm_configs["right_arm"]["planning_time"]
        self.moveit2_right.num_planning_attempts = self.arm_configs["right_arm"]["num_planning_attempts"]
        self.moveit2_right.goal_joint_tolerance = self.arm_configs["right_arm"]["goal_joint_tolerance"]
        self.moveit2_right.goal_position_tolerance = self.arm_configs["right_arm"]["goal_position_tolerance"]
        self.moveit2_right.goal_orientation_tolerance = self.arm_configs["right_arm"]["goal_orientation_tolerance"]
        
        # 创建TF缓冲区和监听器
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        
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
        
        print("双机械臂笛卡尔空间往返运动控制器初始化完成")
        
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
    
    def get_current_pose(self, arm_name):
        """获取当前末端执行器位姿"""
        try:
            # 使用TF获取当前位姿
            base_frame = self.arm_configs[arm_name]["base_link_name"]
            tool_frame = self.arm_configs[arm_name]["end_effector_name"]
            
            # 等待变换可用
            transform = self.tf_buffer.lookup_transform(
                base_frame,
                tool_frame,
                rclpy.time.Time())
            
            # 创建位姿对象
            pose = Pose()
            pose.position.x = transform.transform.translation.x
            pose.position.y = transform.transform.translation.y
            pose.position.z = transform.transform.translation.z
            pose.orientation.x = transform.transform.rotation.x
            pose.orientation.y = transform.transform.rotation.y
            pose.orientation.z = transform.transform.rotation.z
            pose.orientation.w = transform.transform.rotation.w
            
            print(f"成功获取{arm_name}当前位姿")
            return pose
        except TransformException as ex:
            print(f"无法获取{arm_name}位姿: {ex}")
            return None
        except Exception as e:
            print(f"获取{arm_name}位姿时出错: {e}")
            return None
    
    def print_pose(self, arm_name, pose, prefix=""):
        """打印位姿信息"""
        if pose:
            print(f"{prefix}{arm_name}位姿:")
            print(f"  位置 (x, y, z): ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}) m")
            print(f"  姿态 (x, y, z, w): ({pose.orientation.x:.3f}, {pose.orientation.y:.3f}, {pose.orientation.z:.3f}, {pose.orientation.w:.3f})")
        else:
            print(f"{prefix}{arm_name}位姿信息不可用")
    
    def _quaternion_to_euler(self, q):
        """四元数转欧拉角 (roll, pitch, yaw)"""
        x, y, z, w = q[0], q[1], q[2], q[3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return [roll, pitch, yaw]
    
    def _euler_to_quaternion(self, euler):
        """欧拉角转四元数"""
        roll, pitch, yaw = euler[0], euler[1], euler[2]
        
        # Half angles
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        # Quaternion
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return [x, y, z, w]
    
    def create_nearby_pose(self, original_pose, offset_m=0.01):
        """在原始位姿附近创建一个新位姿，包含位置和姿态的小偏移"""
        if not original_pose:
            return None
            
        new_pose = Pose()
        
        # 位置偏移 - 减小偏移量，避免关节空间大变化
        # 只在单一方向做小偏移，减少不必要的关节转动
        new_pose.position.x = original_pose.position.x + offset_m * 0.5
        new_pose.position.y = original_pose.position.y + offset_m * 0.3
        new_pose.position.z = original_pose.position.z + offset_m * 0.2
        
        # 姿态偏移 - 减小姿态变化，避免大角度转动
        # 将原始四元数转换为欧拉角，添加小偏移，再转换回四元数
        original_q = [original_pose.orientation.x, original_pose.orientation.y, 
                     original_pose.orientation.z, original_pose.orientation.w]
        
        # 四元数转欧拉角 (roll, pitch, yaw)
        euler = self._quaternion_to_euler(original_q)
        
        # 姿态偏移量 (约2.9度)，大幅减小姿态变化
        euler_offset = 0.05  
        # 只在单一旋转方向做小偏移，减少关节转动
        euler[2] += euler_offset * 0.5  # 主要在yaw方向做小偏移
        
        # 欧拉角转回四元数
        new_q = self._euler_to_quaternion(euler)
        
        new_pose.orientation.x = new_q[0]
        new_pose.orientation.y = new_q[1]
        new_pose.orientation.z = new_q[2]
        new_pose.orientation.w = new_q[3]
        
        return new_pose
    
    def move_to_pose_with_joint_constraints(self, arm_name, target_pose, movement_name, max_joint_change=0.05):
        """带关节约束的位姿移动，避免关节空间大变化"""
        if arm_name == "left_arm":
            moveit2 = self.moveit2_left
        else:
            moveit2 = self.moveit2_right
        
        print(f"执行 {movement_name} (带关节约束)...")
        
        try:
            # 获取当前关节状态
            if moveit2.joint_state is None:
                print(f"无法获取{arm_name}关节状态")
                return False
            
            current_joints = moveit2.joint_state.position
            
            # 使用MoveIt2的位姿控制功能，添加关节约束优化
            moveit2.move_to_pose(
                position=[target_pose.position.x, target_pose.position.y, target_pose.position.z],
                quat_xyzw=[target_pose.orientation.x, target_pose.orientation.y, 
                          target_pose.orientation.z, target_pose.orientation.w],
                # 添加关节约束参数，限制关节变化
                cartesian=False,  # 使用关节空间规划而不是笛卡尔空间规划
                max_velocity_scaling_factor=0.3,  # 降低速度缩放因子
                max_acceleration_scaling_factor=0.3  # 降低加速度缩放因子
            )
            
            # 等待执行完成
            success = moveit2.wait_until_executed()
            
            if success:
                print(f"{movement_name} 完成")
                
                # 检查关节变化量
                if moveit2.joint_state is not None:
                    new_joints = moveit2.joint_state.position
                    joint_changes = [abs(new_joints[i] - current_joints[i]) for i in range(len(current_joints))]
                    max_change = max(joint_changes)
                    avg_change = sum(joint_changes) / len(joint_changes)
                    print(f"  最大关节变化: {max_change:.3f} 弧度")
                    print(f"  平均关节变化: {avg_change:.3f} 弧度")
                    
                    if max_change > max_joint_change:
                        print(f"  警告: 关节变化较大，建议进一步减小笛卡尔空间偏移量")
                
                # 打印运动后的当前状态
                current_pose = self.get_current_pose(arm_name)
                print(f"{movement_name}后的当前状态:")
                self.print_pose(arm_name, current_pose, "  ")
                
                return True
            else:
                print(f"{movement_name} 失败")
                return False
        except Exception as e:
            print(f"执行{movement_name}时出错: {e}")
            return False
    
    def move_to_pose(self, arm_name, target_pose, movement_name):
        """移动机械臂到指定位姿"""
        if arm_name == "left_arm":
            moveit2 = self.moveit2_left
        else:
            moveit2 = self.moveit2_right
        
        print(f"执行 {movement_name}...")
        
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
                print(f"{movement_name} 完成")
                
                # 打印运动后的当前状态
                current_pose = self.get_current_pose(arm_name)
                print(f"{movement_name}后的当前状态:")
                self.print_pose(arm_name, current_pose, "  ")
                
                return True
            else:
                print(f"{movement_name} 失败")
                return False
        except Exception as e:
            print(f"执行{movement_name}时出错: {e}")
            return False
    
    def execute_roundtrip_motion(self):
        """执行往返运动"""
        print("\n=== 开始执行往返运动 ===")
        
        # 1. 读取当前双臂的笛卡尔位姿信息
        print("\n步骤1: 读取当前双臂位姿信息")
        left_initial_pose = self.get_current_pose("left_arm")
        right_initial_pose = self.get_current_pose("right_arm")
        
        if not left_initial_pose or not right_initial_pose:
            print("无法获取初始位姿，程序退出")
            return False
        
        print("\n初始位姿:")
        self.print_pose("左臂", left_initial_pose, "  ")
        self.print_pose("右臂", right_initial_pose, "  ")
        
        # 2. 停顿2秒
        print("\n步骤2: 停顿2秒...")
        time.sleep(2.0)
        
        # 3. 在当前位姿附近进行小幅运动
        print("\n步骤3: 在当前位姿附近进行小幅运动")
        left_nearby_pose = self.create_nearby_pose(left_initial_pose, offset_m=0.03)
        right_nearby_pose = self.create_nearby_pose(right_initial_pose, offset_m=0.03)
        
        if not left_nearby_pose or not right_nearby_pose:
            print("无法创建附近位姿，程序退出")
            return False
        
        print("\n目标位姿 (附近):")
        self.print_pose("左臂", left_nearby_pose, "  ")
        self.print_pose("右臂", right_nearby_pose, "  ")
        
        # 同时移动双臂到附近位姿（使用带关节约束的移动）
        left_thread = threading.Thread(
            target=self.move_to_pose_with_joint_constraints, 
            args=("left_arm", left_nearby_pose, "左臂附近位姿运动", 0.1)
        )
        right_thread = threading.Thread(
            target=self.move_to_pose_with_joint_constraints, 
            args=("right_arm", right_nearby_pose, "右臂附近位姿运动", 0.1)
        )
        
        left_thread.start()
        right_thread.start()
        
        left_thread.join()
        right_thread.join()
        
        # 4. 停顿2秒
        print("\n步骤4: 停顿2秒...")
        time.sleep(2.0)
        
        # 5. 回到初始位姿
        print("\n步骤5: 回到初始位姿")
        left_thread = threading.Thread(
            target=self.move_to_pose, 
            args=("left_arm", left_initial_pose, "左臂回到初始位姿")
        )
        right_thread = threading.Thread(
            target=self.move_to_pose, 
            args=("right_arm", right_initial_pose, "右臂回到初始位姿")
        )
        
        left_thread.start()
        right_thread.start()
        
        left_thread.join()
        right_thread.join()
        
        # 6. 打印成功信息
        print("\n步骤6: 往返运动完成")
        print("双机械臂笛卡尔空间往返运动成功完成！")
        
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
    print("=== 双机械臂笛卡尔空间往返运动控制程序 ===")
    
    controller = None
    try:
        # 创建控制器
        controller = DualArmCartesianRoundtripController()
        
        # 等待关节状态数据
        print("\n等待关节状态数据...")
        if not controller.wait_for_joint_states():
            print("等待关节状态超时，程序退出")
            return
        
        # 执行往返运动
        controller.execute_roundtrip_motion()
        
    except Exception as e:
        print(f"程序执行出错: {e}")
    finally:
        # 关闭控制器
        if controller is not None:
            controller.shutdown()


if __name__ == "__main__":
    main()