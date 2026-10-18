#!/usr/bin/env python3
"""
双机械臂轨迹控制程序
控制两只机械臂沿着预定义轨迹连续运动
支持轨迹起点检测和紧急停止功能
"""

import time
import threading
import signal
import sys
from threading import Thread
import math

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from pymoveit2 import MoveIt2
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class DualArmTrajectoryController:
    def __init__(self):
        rclpy.init()
        
        # 创建节点
        self.node = Node("dual_arm_trajectory_controller")
        
        # 创建回调组
        self.callback_group = ReentrantCallbackGroup()
        
        # 紧急停止标志
        self.emergency_stop = False
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # 机械臂配置（增强避障功能）
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
                # 避障相关参数
                "planning_time": 5.0,  # 增加规划时间以提高避障成功率
                "num_planning_attempts": 10,  # 增加规划尝试次数
                "allowed_planning_time": 10.0,  # 允许的最大规划时间
                "replan": True,  # 允许重新规划
                "replan_attempts": 3,  # 重新规划尝试次数
                "replan_delay": 2.0  # 重新规划延迟
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
                # 避障相关参数
                "planning_time": 5.0,
                "num_planning_attempts": 10,
                "allowed_planning_time": 10.0,
                "replan": True,
                "replan_attempts": 3,
                "replan_delay": 2.0
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
        
        # 设置避障相关参数
        self.setup_collision_avoidance()
        
        # 启动执行器
        self.executor = MultiThreadedExecutor(2)
        self.executor.add_node(self.node)
        self.executor_thread = Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        
        # 等待初始化
        print("等待MoveIt2初始化...")
        for i in range(5):
            try:
                self.node.create_rate(1.0).sleep()
                print(f"初始化进度: {i+1}/5")
            except Exception as e:
                print(f"初始化等待出错: {e}")
                break
        
        print("双机械臂轨迹控制器初始化完成")
    
    def signal_handler(self, signum, frame):
        """信号处理函数，处理Ctrl+C"""
        print("\n收到停止信号，执行紧急停止...")
        self.emergency_stop = True
        self.emergency_stop_motion()
    
    def emergency_stop_motion(self):
        """紧急停止机械臂运动"""
        print("执行紧急停止...")
        try:
            # 停止左臂运动
            self.moveit2_left.stop()
            print("左臂已停止")
        except Exception as e:
            print(f"停止左臂时出错: {e}")
        
        try:
            # 停止右臂运动
            self.moveit2_right.stop()
            print("右臂已停止")
        except Exception as e:
            print(f"停止右臂时出错: {e}")
    
    def setup_collision_avoidance(self):
        """设置碰撞避免参数"""
        print("设置碰撞避免参数...")
        
        # 设置左臂避障参数
        try:
            # 设置规划时间
            self.moveit2_left.planning_time = self.arm_configs["left_arm"]["planning_time"]
            # 设置规划尝试次数
            self.moveit2_left.num_planning_attempts = self.arm_configs["left_arm"]["num_planning_attempts"]
            # 设置允许的规划时间
            self.moveit2_left.allowed_planning_time = self.arm_configs["left_arm"]["allowed_planning_time"]
            # 启用重新规划
            self.moveit2_left.replan = self.arm_configs["left_arm"]["replan"]
            self.moveit2_left.replan_attempts = self.arm_configs["left_arm"]["replan_attempts"]
            self.moveit2_left.replan_delay = self.arm_configs["left_arm"]["replan_delay"]
            
            print("左臂避障参数设置完成")
        except Exception as e:
            print(f"设置左臂避障参数时出错: {e}")
        
        # 设置右臂避障参数
        try:
            self.moveit2_right.planning_time = self.arm_configs["right_arm"]["planning_time"]
            self.moveit2_right.num_planning_attempts = self.arm_configs["right_arm"]["num_planning_attempts"]
            self.moveit2_right.allowed_planning_time = self.arm_configs["right_arm"]["allowed_planning_time"]
            self.moveit2_right.replan = self.arm_configs["right_arm"]["replan"]
            self.moveit2_right.replan_attempts = self.arm_configs["right_arm"]["replan_attempts"]
            self.moveit2_right.replan_delay = self.arm_configs["right_arm"]["replan_delay"]
            
            print("右臂避障参数设置完成")
        except Exception as e:
            print(f"设置右臂避障参数时出错: {e}")
        
        print("碰撞避免参数设置完成")
    
    def wait_for_joint_states(self, timeout=10.0):
        """等待关节状态数据可用"""
        start_time = time.time()
        print(f"等待关节状态数据，超时时间: {timeout}秒...")
        
        while time.time() - start_time < timeout:
            if self.emergency_stop:
                print("检测到紧急停止，退出等待")
                return False
                
            left_state = self.moveit2_left.joint_state is not None
            right_state = self.moveit2_right.joint_state is not None
            
            if left_state and right_state:
                print("关节状态数据已就绪")
                return True
            
            elapsed = time.time() - start_time
            print(f"等待中... 已等待: {elapsed:.1f}秒, 左臂状态: {left_state}, 右臂状态: {right_state}")
            time.sleep(1.0)
        
        print(f"等待关节状态超时")
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
    
    def check_pose_similarity(self, current_joints, target_joints, tolerance=0.1):
        """
        检查当前关节位置是否接近目标位置
        
        参数:
            current_joints: 当前关节位置（弧度）
            target_joints: 目标关节位置（弧度）
            tolerance: 允许的误差（弧度）
        
        返回:
            bool: 是否在允许误差范围内
        """
        if len(current_joints) != len(target_joints):
            return False
        
        for i in range(len(current_joints)):
            if abs(current_joints[i] - target_joints[i]) > tolerance:
                return False
        
        return True
    
    def move_to_start_position(self, arm_name, start_joints):
        """移动机械臂到轨迹起点"""
        if arm_name == "left_arm":
            moveit2 = self.moveit2_left
        else:
            moveit2 = self.moveit2_right
        
        print(f"{arm_name} 正在移动到轨迹起点...")
        
        try:
            moveit2.move_to_configuration(start_joints)
            success = moveit2.wait_until_executed()
            
            if success:
                print(f"{arm_name} 已到达轨迹起点")
                return True
            else:
                print(f"{arm_name} 移动到轨迹起点失败")
                return False
        except Exception as e:
            print(f"{arm_name} 移动到轨迹起点时出错: {e}")
            return False
    
    def define_trajectory(self):
        """定义机械臂轨迹点"""
        print("定义双机械臂轨迹...")
        
        # 定义左臂轨迹（示例轨迹，可根据需要修改）
        left_trajectory_points = [
            # 起点位置（关节角度，弧度）
            [0.0, -0.5, 0.0, 1.0, 0.0, -0.5, 0.0],
            # 中间点1
            [0.2, -0.3, 0.1, 1.2, 0.1, -0.4, 0.1],
            # 中间点2
            [0.4, -0.1, 0.2, 1.4, 0.2, -0.3, 0.2],
            # 中间点3
            [0.6, 0.1, 0.3, 1.6, 0.3, -0.2, 0.3],
            # 终点位置
            [0.8, 0.3, 0.4, 1.8, 0.4, -0.1, 0.4]
        ]
        
        # 定义右臂轨迹（镜像对称）
        right_trajectory_points = [
            # 起点位置
            [0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0],
            # 中间点1
            [-0.2, 0.3, -0.1, -1.2, -0.1, 0.4, -0.1],
            # 中间点2
            [-0.4, 0.1, -0.2, -1.4, -0.2, 0.3, -0.2],
            # 中间点3
            [-0.6, -0.1, -0.3, -1.6, -0.3, 0.2, -0.3],
            # 终点位置
            [-0.8, -0.3, -0.4, -1.8, -0.4, 0.1, -0.4]
        ]
        
        return {
            "left_arm": left_trajectory_points,
            "right_arm": right_trajectory_points
        }
    
    def execute_trajectory_point(self, arm_name, target_joints, point_index, total_points):
        """执行单个轨迹点（带避障功能）"""
        if self.emergency_stop:
            print(f"检测到紧急停止，跳过{arm_name}第{point_index+1}个轨迹点")
            return False
        
        if arm_name == "left_arm":
            moveit2 = self.moveit2_left
        else:
            moveit2 = self.moveit2_right
        
        print(f"{arm_name} 执行第{point_index+1}/{total_points}个轨迹点（带避障）...")
        
        try:
            # 使用MoveIt2的关节空间规划（自动避障）
            moveit2.move_to_configuration(target_joints)
            
            # 等待执行完成，带超时处理
            success = moveit2.wait_until_executed()
            
            if success:
                print(f"{arm_name} 第{point_index+1}个轨迹点执行完成（避障成功）")
                return True
            else:
                print(f"{arm_name} 第{point_index+1}个轨迹点执行失败，尝试重新规划...")
                
                # 尝试重新规划
                for attempt in range(2):  # 最多尝试2次重新规划
                    if self.emergency_stop:
                        break
                        
                    print(f"重新规划尝试 {attempt+1}/2...")
                    moveit2.move_to_configuration(target_joints)
                    success = moveit2.wait_until_executed()
                    
                    if success:
                        print(f"重新规划成功，{arm_name} 第{point_index+1}个轨迹点执行完成")
                        return True
                    
                    time.sleep(1.0)  # 等待后重试
                
                print(f"{arm_name} 第{point_index+1}个轨迹点多次规划失败")
                return False
                
        except Exception as e:
            print(f"{arm_name} 执行第{point_index+1}个轨迹点时出错: {e}")
            
            # 检查是否是碰撞相关的错误
            if "collision" in str(e).lower() or "obstacle" in str(e).lower():
                print("检测到碰撞或障碍物，尝试重新规划路径...")
                # 这里可以添加更复杂的避障策略
                
            return False
    
    def execute_trajectory(self):
        """执行完整轨迹"""
        print("开始执行双机械臂轨迹...")
        
        # 定义轨迹
        trajectory = self.define_trajectory()
        
        # 获取轨迹起点
        left_start = trajectory["left_arm"][0]
        right_start = trajectory["right_arm"][0]
        
        # 检查是否需要移动到起点
        print("检查机械臂当前位置...")
        
        left_current = self.get_current_joint_positions("left_arm")
        right_current = self.get_current_joint_positions("right_arm")
        
        if left_current and not self.check_pose_similarity(left_current, left_start):
            print("左臂不在轨迹起点，需要先移动到起点")
            if not self.move_to_start_position("left_arm", left_start):
                print("左臂移动到起点失败，终止程序")
                return False
        else:
            print("左臂已在轨迹起点附近")
        
        if right_current and not self.check_pose_similarity(right_current, right_start):
            print("右臂不在轨迹起点，需要先移动到起点")
            if not self.move_to_start_position("right_arm", right_start):
                print("右臂移动到起点失败，终止程序")
                return False
        else:
            print("右臂已在轨迹起点附近")
        
        # 执行轨迹
        total_points = len(trajectory["left_arm"])
        
        for i in range(total_points):
            if self.emergency_stop:
                print("检测到紧急停止，终止轨迹执行")
                break
            
            print(f"\n=== 执行第{i+1}/{total_points}个轨迹点 ===")
            
            # 执行左臂轨迹点
            left_success = self.execute_trajectory_point("left_arm", trajectory["left_arm"][i], i, total_points)
            
            # 执行右臂轨迹点
            right_success = self.execute_trajectory_point("right_arm", trajectory["right_arm"][i], i, total_points)
            
            if not left_success or not right_success:
                print("轨迹点执行失败，终止程序")
                return False
            
            # 短暂停顿，确保运动稳定
            if i < total_points - 1:  # 最后一个点后不需要停顿
                print("轨迹点间停顿1秒...")
                for j in range(10):  # 分10次检查，每次0.1秒，便于响应紧急停止
                    if self.emergency_stop:
                        break
                    time.sleep(0.1)
        
        if not self.emergency_stop:
            print("\n=== 轨迹执行完成 ===")
            print("双机械臂已到达轨迹终点")
        else:
            print("\n=== 轨迹执行被中断 ===")
        
        return True
    
    def print_joint_angles(self, arm_name, joint_positions, prefix=""):
        """打印关节角度信息"""
        print(f"{prefix}{arm_name}关节角度 (度):")
        for i, angle in enumerate(joint_positions):
            print(f"  关节{i+1}: {angle * 180.0 / 3.141592654:.2f}°")
    
    def shutdown(self):
        """关闭控制器"""
        try:
            if rclpy.ok():
                rclpy.shutdown()
            if self.executor_thread.is_alemon():
                self.executor_thread.join(timeout=1.0)
        except Exception as e:
            print(f"关闭控制器时出错: {e}")


def main():
    controller = None
    try:
        # 创建控制器
        controller = DualArmTrajectoryController()
        
        # 等待关节状态数据
        print("等待关节状态数据...")
        if not controller.wait_for_joint_states():
            print("等待关节状态超时，程序退出")
            return
        
        # 显示初始状态
        print("\n初始状态:")
        left_initial = controller.get_current_joint_positions("left_arm")
        right_initial = controller.get_current_joint_positions("right_arm")
        
        if left_initial:
            controller.print_joint_angles("左臂", left_initial, "初始")
        if right_initial:
            controller.print_joint_angles("右臂", right_initial, "初始")
        
        # 执行轨迹
        success = controller.execute_trajectory()
        
        # 显示最终状态
        if not controller.emergency_stop:
            print("\n最终状态:")
            left_final = controller.get_current_joint_positions("left_arm")
            right_final = controller.get_current_joint_positions("right_arm")
            
            if left_final:
                controller.print_joint_angles("左臂", left_final, "最终")
            if right_final:
                controller.print_joint_angles("右臂", right_final, "最终")
        
        if success and not controller.emergency_stop:
            print("\n双机械臂轨迹控制程序执行完成！")
        else:
            print("\n程序执行异常结束")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
    finally:
        # 关闭控制器
        if controller is not None:
            controller.shutdown()


if __name__ == "__main__":
    main()