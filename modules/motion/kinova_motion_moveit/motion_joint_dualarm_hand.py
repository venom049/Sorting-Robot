#!/usr/bin/env python3
"""
双机械臂与灵巧手联动控制（循环直到 Ctrl+C）
通过调用已有的 MoveIt2 接口控制左右臂，并在停留阶段调用 linker_hand_ros2_sdk 控制 L20 灵巧手
"""
import time
import threading
import signal
import sys

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from sensor_msgs.msg import JointState

from pymoveit2 import MoveIt2

# 尝试导入 linker hand SDK（假定已在工作空间中并安装）
try:
    from linker_hand_ros2_sdk.linker_hand import LinkerHand
except ImportError as e:
    LinkerHand = None
    print(f"无法导入 LinkerHand: {e}")
    print("请确保 linker_hand_ros2_sdk 包已正确安装")
except Exception as e:
    LinkerHand = None
    print(f"导入 LinkerHand 时发生错误: {e}")


class DualArmHandController:
    def __init__(self):
        rclpy.init()
        self.node = Node("motion_joint_dualarm_hand")

        self.callback_group = ReentrantCallbackGroup()

        # 关节配置与 MoveIt2 接口（添加速度限制参数）
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
                "max_velocity": 0.05,  # 降低最大速度
                "max_acceleration": 0.02,  # 降低最大加速度
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
                "max_velocity": 0.05,  # 降低最大速度
                "max_acceleration": 0.02,  # 降低最大加速度
            }
        }

        self.moveit2_left = MoveIt2(
            node=self.node,
            joint_names=self.arm_configs["left_arm"]["joint_names"],
            base_link_name=self.arm_configs["left_arm"]["base_link_name"],
            end_effector_name=self.arm_configs["left_arm"]["end_effector_name"],
            group_name=self.arm_configs["left_arm"]["group_name"],
            callback_group=self.callback_group,
        )

        self.moveit2_right = MoveIt2(
            node=self.node,
            joint_names=self.arm_configs["right_arm"]["joint_names"],
            base_link_name=self.arm_configs["right_arm"]["base_link_name"],
            end_effector_name=self.arm_configs["right_arm"]["end_effector_name"],
            group_name=self.arm_configs["right_arm"]["group_name"],
            callback_group=self.callback_group,
        )

        # 设置规划器参数（速度限制）
        self.moveit2_left.planner_id = self.arm_configs["left_arm"]["planner_id"]
        self.moveit2_left.max_velocity = self.arm_configs["left_arm"]["max_velocity"]
        self.moveit2_left.max_acceleration = self.arm_configs["left_arm"]["max_acceleration"]
        
        self.moveit2_right.planner_id = self.arm_configs["right_arm"]["planner_id"]
        self.moveit2_right.max_velocity = self.arm_configs["right_arm"]["max_velocity"]
        self.moveit2_right.max_acceleration = self.arm_configs["right_arm"]["max_acceleration"]

        self.executor = MultiThreadedExecutor(2)
        self.executor.add_node(self.node)
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

        # linker hand client - 使用 ROS2 主题通信方式
        self.hand = None
        if LinkerHand is not None:
            try:
                # 创建灵巧手控制发布器
                self.hand_cmd_pub = self.node.create_publisher(JointState, '/cb_left_hand_control_cmd', 10)
                self.node.get_logger().info("灵巧手控制发布器创建成功")
            except Exception as e:
                self.node.get_logger().warn(f"初始化灵巧手控制失败: {e}")
        else:
            self.node.get_logger().warn("无法导入 linker_hand_ros2_sdk.linker_hand，灵巧手功能将被跳过（请确保已安装该 SDK）")

        # 用于优雅退出
        self._running = True

    def shutdown(self):
        self._running = False
        rclpy.shutdown()
        self.executor_thread.join()

    def wait_for_joint_states(self, timeout=10.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if (self.moveit2_left.joint_state is not None and self.moveit2_right.joint_state is not None):
                return True
            time.sleep(0.1)
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
                idx = moveit2.joint_state.name.index(joint_name)
                positions.append(moveit2.joint_state.position[idx])
            else:
                positions.append(0.0)
        return positions

    def move_joints(self, arm_name, target_positions_deg):
        if arm_name == "left_arm":
            moveit2 = self.moveit2_left
        else:
            moveit2 = self.moveit2_right

        current_rad = self.get_current_joint_positions(arm_name)
        if current_rad is None:
            self.node.get_logger().warn(f"{arm_name} 关节状态不可用")
            return False

        current_deg = [p * 180.0 / 3.141592654 for p in current_rad]
        final_rad = []
        for i, target_deg in enumerate(target_positions_deg):
            if target_deg is None:
                final_rad.append(current_rad[i])
            else:
                final_rad.append(target_deg * 3.141592654 / 180.0)

        self.node.get_logger().info(f"{arm_name} 从 {[f'{p:.1f}°' for p in current_deg]} 移动到 {[f'{p:.1f}°' for p in target_positions_deg]}")
        moveit2.move_to_configuration(final_rad)
        return moveit2.wait_until_executed()

    def call_hand_position(self, positions):
        """调用灵巧手 SDK 将手指移动到 positions（列表或可迭代）"""
        if self.hand is None:
            self.node.get_logger().info("未启用灵巧手客户端，跳过手部动作")
            return False
        try:
            # 通过 ROS2 主题发送灵巧手控制指令
            if hasattr(self, 'hand_cmd_pub'):
                msg = JointState()
                msg.header.stamp = self.node.get_clock().now().to_msg()
                msg.name = [f'finger_{i}' for i in range(len(positions))]
                msg.position = positions
                self.hand_cmd_pub.publish(msg)
                self.node.get_logger().info(f"发送灵巧手控制指令: {positions}")
                time.sleep(0.2)
                return True
            else:
                self.node.get_logger().info(f"灵巧手控制指令: {positions}")
                time.sleep(0.2)
                return True
        except Exception as e:
            self.node.get_logger().warn(f"发送灵巧手指令失败: {e}")
            return False

    def calibrate_joint_positions(self):
        """校准关节位置，读取当前机械臂的实际位置"""
        self.node.get_logger().info("开始校准关节位置...")
        
        # 等待关节状态稳定
        time.sleep(0.5)
        
        left_rad = self.get_current_joint_positions("left_arm")
        right_rad = self.get_current_joint_positions("right_arm")
        
        if left_rad is None or right_rad is None:
            self.node.get_logger().error("校准失败：无法读取关节位置")
            return None, None
        
        left_deg = [p * 180.0 / 3.141592654 for p in left_rad]
        right_deg = [p * 180.0 / 3.141592654 for p in right_rad]
        
        self.node.get_logger().info(f"校准完成：左臂位置 {[f'{p:.1f}°' for p in left_deg]}")
        self.node.get_logger().info(f"校准完成：右臂位置 {[f'{p:.1f}°' for p in right_deg]}")
        
        return left_deg, right_deg

    def run_loop(self):
        # 等待 joint_states
        if not self.wait_for_joint_states():
            self.node.get_logger().error("等待 joint_states 超时")
            return

        # 初始校准
        left_deg, right_deg = self.calibrate_joint_positions()
        if left_deg is None or right_deg is None:
            self.node.get_logger().error("初始校准失败，程序退出")
            return

        # 预定义的灵巧手位置
        hand_pos_open = [30,68,60,66,54,188,108,150,152,180,32,0,0,0,0,50,100,90,50,60]
        hand_pos_close = [252,230,229,229,229,0,190,189,190,190,146,0,0,0,0,254,255,254,253,254]

        self.node.get_logger().info("进入循环，按 Ctrl+C 停止")
        
        cycle_count = 0

        while self._running:
            # 每10次循环校准一次位置
            if cycle_count % 10 == 0 and cycle_count > 0:
                self.node.get_logger().info(f"已完成 {cycle_count} 次循环，开始校准...")
                new_left_deg, new_right_deg = self.calibrate_joint_positions()
                if new_left_deg is not None and new_right_deg is not None:
                    left_deg = new_left_deg
                    right_deg = new_right_deg
                    self.node.get_logger().info("位置校准完成，继续循环")
                else:
                    self.node.get_logger().warn("位置校准失败，使用上次位置继续")

            # 增加5度
            left_plus = [p + 5 for p in left_deg]
            right_plus = [p + 5 for p in right_deg]

            t_left = threading.Thread(target=self.move_joints, args=("left_arm", left_plus))
            t_right = threading.Thread(target=self.move_joints, args=("right_arm", right_plus))
            t_left.start(); t_right.start()
            t_left.join(); t_right.join()

            # 停留并控制灵巧手到 hand_pos_open
            self.call_hand_position(hand_pos_open)
            time.sleep(2)

            # 回到校准位置
            t_left = threading.Thread(target=self.move_joints, args=("left_arm", left_deg))
            t_right = threading.Thread(target=self.move_joints, args=("right_arm", right_deg))
            t_left.start(); t_right.start()
            t_left.join(); t_right.join()

            # 灵巧手回到 hand_pos_close
            self.call_hand_position(hand_pos_close)
            time.sleep(2)
            
            cycle_count += 1
            self.node.get_logger().info(f"已完成循环次数: {cycle_count}")

        self.node.get_logger().info("循环退出，准备关闭")


def main():
    controller = DualArmHandController()

    def shutdown_handler(signum, frame):
        controller.node.get_logger().info("收到终止信号，正在退出...")
        controller.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        controller.run_loop()
    except KeyboardInterrupt:
        controller.shutdown()


if __name__ == '__main__':
    main()
