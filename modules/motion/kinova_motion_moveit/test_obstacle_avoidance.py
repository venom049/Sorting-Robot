#!/usr/bin/env python3
"""
双机械臂避障控制测试版本
简化版本，使用MoveIt内置的避障功能，确保基本功能正常
"""

import time
import threading
import signal
import sys
import math
from threading import Thread

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from pymoveit2 import MoveIt2
from geometry_msgs.msg import Pose


class SimpleObstacleAvoidanceController:
    """简化的避障控制器，使用MoveIt内置功能"""
    
    def __init__(self):
        rclpy.init()
        
        # 创建节点
        self.node = Node("simple_obstacle_avoidance_controller")
        
        # 创建回调组
        self.callback_group = ReentrantCallbackGroup()
        
        # 紧急停止标志
        self.emergency_stop = False
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        
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
                "max_velocity": 0.03,
                "max_acceleration": 0.01
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
                "max_velocity": 0.03,
                "max_acceleration": 0.01
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
        for i in range(5):
            try:
                self.node.create_rate(1.0).sleep()
                print(f"初始化进度: {i+1}/5")
            except Exception as e:
                print(f"初始化等待出错: {e}")
                break
        
        print("简化避障控制器初始化完成")
    
    def signal_handler(self, signum, frame):
        """信号处理函数"""
        print("\n收到停止信号，执行紧急停止...")
        self.emergency_stop = True
        self.emergency_stop_motion()
    
    def emergency_stop_motion(self):
        """紧急停止机械臂运动"""
        print("执行紧急停止...")
        try:
            # 设置极低的速度来快速停止
            self.moveit2_left.max_velocity = 0.001
            self.moveit2_right.max_velocity = 0.001
            print("已降低速度进行紧急停止")
        except Exception as e:
            print(f"紧急停止时出错: {e}")
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """欧拉角转四元数"""
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
        
        return w, x, y, z
    
    def create_pose(self, x, y, z, roll=0, pitch=0, yaw=0):
        """创建位姿"""
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        
        w, qx, qy, qz = self.euler_to_quaternion(roll, pitch, yaw)
        pose.orientation.w = w
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        
        return pose
    
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
    
    def move_arm_to_pose(self, arm_name, target_pose):
        """移动机械臂到目标位姿（使用MoveIt内置避障）"""
        print(f"\n=== 开始{arm_name}避障运动 ===")
        print(f"目标位置: ({target_pose.position.x:.3f}, "
              f"{target_pose.position.y:.3f}, {target_pose.position.z:.3f})")
        
        if arm_name == "left_arm":
            moveit2 = self.moveit2_left
        else:
            moveit2 = self.moveit2_right
        
        try:
            print(f"开始{arm_name}运动规划...")
            
            # 使用MoveIt的move_to_pose，它会自动进行避障规划
            moveit2.move_to_pose(target_pose)
            
            print(f"等待{arm_name}运动完成...")
            success = moveit2.wait_until_executed()
            
            if success:
                print(f"{arm_name}成功到达目标位姿")
                return True
            else:
                print(f"{arm_name}运动失败")
                return False
                
        except Exception as e:
            print(f"{arm_name}运动过程中出错: {e}")
            return False
    
    def move_both_arms_to_poses(self, left_target_pose, right_target_pose):
        """同时控制双臂运动"""
        print("\n=== 开始双臂协同运动 ===")
        
        # 创建线程同时执行
        left_thread = Thread(
            target=self.move_arm_to_pose,
            args=("left_arm", left_target_pose)
        )
        
        right_thread = Thread(
            target=self.move_arm_to_pose,
            args=("right_arm", right_target_pose)
        )
        
        # 启动线程
        left_thread.start()
        right_thread.start()
        
        # 等待完成
        left_thread.join()
        right_thread.join()
        
        print("=== 双臂协同运动完成 ===")
    
    def shutdown(self):
        """关闭控制器"""
        print("关闭控制器...")
        try:
            self.emergency_stop_motion()
            rclpy.shutdown()
            if self.executor_thread.is_alive():
                self.executor_thread.join(timeout=2.0)
            print("控制器已关闭")
        except Exception as e:
            print(f"关闭控制器时出错: {e}")


def main():
    """主函数"""
    print("=== 双机械臂简化避障控制测试程序 ===")
    print("使用MoveIt内置避障功能")
    print("支持Ctrl+C紧急停止")
    print()
    
    # 创建控制器
    controller = SimpleObstacleAvoidanceController()
    
    try:
        # 等待关节状态数据
        if not controller.wait_for_joint_states():
            print("等待关节状态超时，程序退出")
            return
        
        # 测试1: 左臂运动
        print("\n--- 测试1: 左臂运动 ---")
        left_target = controller.create_pose(
            x=0.4, y=0.2, z=0.5,
            roll=0, pitch=math.pi/6, yaw=0
        )
        
        success = controller.move_arm_to_pose("left_arm", left_target)
        print(f"左臂运动结果: {'成功' if success else '失败'}")
        
        time.sleep(2.0)
        
        # 测试2: 右臂运动
        print("\n--- 测试2: 右臂运动 ---")
        right_target = controller.create_pose(
            x=-0.35, y=-0.2, z=0.5,
            roll=0, pitch=-math.pi/6, yaw=0
        )
        
        success = controller.move_arm_to_pose("right_arm", right_target)
        print(f"右臂运动结果: {'成功' if success else '失败'}")
        
        time.sleep(2.0)
        
        # 测试3: 双臂协同运动
        print("\n--- 测试3: 双臂协同运动 ---")
        left_target = controller.create_pose(
            x=0.3, y=0.25, z=0.4,
            roll=0, pitch=math.pi/8, yaw=math.pi/4
        )
        
        right_target = controller.create_pose(
            x=-0.3, y=-0.25, z=0.4,
            roll=0, pitch=-math.pi/8, yaw=-math.pi/4
        )
        
        controller.move_both_arms_to_poses(left_target, right_target)
        
        print("\n=== 所有测试完成 ===")
        
    except KeyboardInterrupt:
        print("\n收到键盘中断信号")
    except Exception as e:
        print(f"程序执行过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭控制器
        controller.shutdown()


if __name__ == "__main__":
    main()