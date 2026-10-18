#!/usr/bin/env python3
"""
双机械臂关节运动控制程序
控制左臂和右臂同时进行关节运动
"""

import time
import threading
from threading import Thread

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from pymoveit2 import MoveIt2


class DualArmController:
    def __init__(self):
        rclpy.init()
        
        # 创建节点
        self.node = Node("dual_arm_controller")
        
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
        self.node.create_rate(1.0).sleep()
        
        print("双机械臂控制器初始化完成")
    
    def wait_for_joint_states(self, timeout=10.0):
        """等待关节状态数据可用"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if (self.moveit2_left.joint_state is not None and 
                self.moveit2_right.joint_state is not None):
                return True
            time.sleep(0.1)
        
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
    
    def move_joints(self, arm_name, target_positions_deg):
        """移动机械臂到指定关节位置（角度）"""
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
        
        # 构建最终目标位置（弧度）
        final_positions_rad = []
        for i, (target_deg, current_deg) in enumerate(zip(target_positions_deg, current_positions_deg)):
            if target_deg is not None:
                # 使用目标角度值
                final_positions_rad.append(target_deg * 3.141592654 / 180.0)
            else:
                # 使用当前位置
                final_positions_rad.append(current_positions_rad[i])
        
        print(f"{arm_name} 当前角度: {[f'{p:.1f}°' for p in current_positions_deg]}")
        print(f"{arm_name} 目标角度: {[f'{p:.1f}°' for p in target_positions_deg]}")
        print(f"{arm_name} 移动到关节位置: {[f'{p*180/3.14159:.1f}°' for p in final_positions_rad]}")
        
        # 执行运动
        moveit2.move_to_configuration(final_positions_rad)
        return moveit2.wait_until_executed()
    
    def move_both_arms_simultaneously(self, left_target_deg, right_target_deg):
        """同时移动两只机械臂"""
        left_thread = threading.Thread(target=self.move_joints, args=("left_arm", left_target_deg))
        right_thread = threading.Thread(target=self.move_joints, args=("right_arm", right_target_deg))
        
        left_thread.start()
        right_thread.start()
        
        left_thread.join()
        right_thread.join()
    
    def shutdown(self):
        """关闭控制器"""
        rclpy.shutdown()
        self.executor_thread.join()


def main():
    print("=== 双机械臂关节运动控制程序 ===")
    
    # 创建控制器
    controller = DualArmController()
    
    try:
        # 等待关节状态数据
        print("等待关节状态数据...")
        if not controller.wait_for_joint_states():
            print("等待关节状态超时，程序退出")
            return
        
        # 获取初始关节位置（弧度）
        left_current_rad = controller.get_current_joint_positions("left_arm")
        right_current_rad = controller.get_current_joint_positions("right_arm")
        
        if left_current_rad and right_current_rad:
            # 转换为角度用于显示和计算
            left_current_deg = [p * 180.0 / 3.141592654 for p in left_current_rad]
            right_current_deg = [p * 180.0 / 3.141592654 for p in right_current_rad]
            
            print(f"左臂初始位置: {[f'{p:.1f}°' for p in left_current_deg]}")
            print(f"右臂初始位置: {[f'{p:.1f}°' for p in right_current_deg]}")
        
        # 演示：各关节增加5度
        print("\n1. 两只机械臂各关节增加5度...")
        
        # 创建目标位置（所有关节增加5度）- 使用角度值
        left_target_plus_5 = [p + 5 for p in left_current_deg]
        right_target_plus_5 = [p + 5 for p in right_current_deg]
        
        print(f"左臂目标位置: {[f'{p:.1f}°' for p in left_target_plus_5]}")
        print(f"右臂目标位置: {[f'{p:.1f}°' for p in right_target_plus_5]}")
        
        # 同时移动两只机械臂
        controller.move_both_arms_simultaneously(left_target_plus_5, right_target_plus_5)
        
        # 停留2秒
        print("停留2秒...")
        time.sleep(2)
        
        # 演示：各关节减少5度（回到初始位置）
        print("\n2. 两只机械臂各关节减少5度（回到初始位置）...")
        
        # 同时移动两只机械臂回到初始位置
        controller.move_both_arms_simultaneously(left_current_deg, right_current_deg)
        
        # 停留2秒
        print("停留2秒...")
        time.sleep(2)
        
        print("\n演示完成！")
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {e}")
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()