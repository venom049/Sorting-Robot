#!/usr/bin/env python3
"""
双机械臂自主避障位姿控制程序
使用改进的RRT*算法进行路径规划，实现自主避障运动到目标位姿
支持动态障碍物检测和实时路径重规划
"""

import time
import threading
import signal
import sys
import math
import random
import numpy as np
from collections import deque
from threading import Thread, Lock

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from pymoveit2 import MoveIt2
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


class RRTStarNode:
    """RRT*算法节点类"""
    def __init__(self, position, parent=None):
        self.position = np.array(position)
        self.parent = parent
        self.children = []
        self.cost = 0.0 if parent is None else parent.cost + np.linalg.norm(self.position - parent.position)
    
    def distance_to(self, other):
        """计算到另一个节点或点的距离"""
        if hasattr(other, 'position'):
            # other是RRTStarNode对象
            return np.linalg.norm(self.position - other.position)
        else:
            # other是numpy数组或列表
            return np.linalg.norm(self.position - np.array(other))
    
    def path_to_parent(self):
        """获取到父节点的路径"""
        path = []
        current = self
        while current.parent is not None:
            path.append(current.position.copy())
            current = current.parent
        path.append(current.position.copy())
        return list(reversed(path))


class RRTStarPlanner:
    """改进的RRT*路径规划器"""
    
    def __init__(self, workspace_bounds, max_iterations=5000, step_size=0.05, goal_bias=0.1, 
                 rewire_radius=0.2, obstacle_check_resolution=0.01):
        self.workspace_bounds = workspace_bounds  # [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.rewire_radius = rewire_radius
        self.obstacle_check_resolution = obstacle_check_resolution
        
        self.nodes = []
        self.goal_node = None
        self.best_path = None
        self.best_cost = float('inf')
        
        # 障碍物表示（简化为球体列表）
        self.obstacles = []
        
    def add_obstacle(self, center, radius):
        """添加球形障碍物"""
        self.obstacles.append({'center': np.array(center), 'radius': radius})
    
    def clear_obstacles(self):
        """清空障碍物"""
        self.obstacles = []
    
    def is_collision_free(self, point):
        """检查点是否无碰撞"""
        for obstacle in self.obstacles:
            if np.linalg.norm(point - obstacle['center']) < obstacle['radius']:
                return False
        return True
    
    def is_path_collision_free(self, start, end):
        """检查路径段是否无碰撞"""
        distance = np.linalg.norm(end - start)
        num_checks = int(distance / self.obstacle_check_resolution) + 1
        
        for i in range(num_checks + 1):
            t = i / max(num_checks, 1)
            point = start + t * (end - start)
            if not self.is_collision_free(point):
                return False
        return True
    
    def sample_random_point(self):
        """在工作空间内随机采样点"""
        point = np.array([
            random.uniform(self.workspace_bounds[0][0], self.workspace_bounds[0][1]),
            random.uniform(self.workspace_bounds[1][0], self.workspace_bounds[1][1]),
            random.uniform(self.workspace_bounds[2][0], self.workspace_bounds[2][1])
        ])
        return point
    
    def find_nearest_node(self, point):
        """找到最近的节点"""
        if not self.nodes:
            return None
        
        distances = [node.distance_to(point) for node in self.nodes]
        nearest_index = np.argmin(distances)
        return self.nodes[nearest_index]
    
    def steer(self, from_node, to_point):
        """从节点向目标点扩展"""
        direction = to_point - from_node.position
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return to_point
        
        direction_normalized = direction / distance
        new_point = from_node.position + direction_normalized * self.step_size
        return new_point
    
    def find_near_nodes(self, point, radius):
        """找到半径内的所有节点"""
        near_nodes = []
        for node in self.nodes:
            if node.distance_to(point) <= radius:
                near_nodes.append(node)
        return near_nodes
    
    def choose_parent(self, new_point, near_nodes):
        """为新点选择最优父节点"""
        best_parent = None
        best_cost = float('inf')
        
        for node in near_nodes:
            if self.is_path_collision_free(node.position, new_point):
                cost = node.cost + node.distance_to(new_point)
                if cost < best_cost:
                    best_cost = cost
                    best_parent = node
        
        return best_parent, best_cost
    
    def rewire(self, new_node, near_nodes):
        """重新连接节点以优化路径"""
        for node in near_nodes:
            if node == new_node or node.parent is None:
                continue
            
            if self.is_path_collision_free(new_node.position, node.position):
                new_cost = new_node.cost + new_node.distance_to(node.position)
                if new_cost < node.cost:
                    # 重新连接
                    if node.parent:
                        node.parent.children.remove(node)
                    node.parent = new_node
                    new_node.children.append(node)
                    node.cost = new_cost
    
    def plan(self, start_point, goal_point, goal_tolerance=0.05):
        """执行RRT*路径规划"""
        print(f"开始RRT*路径规划: 起点 {start_point}, 终点 {goal_point}")
        
        # 初始化
        self.nodes = []
        self.best_path = None
        self.best_cost = float('inf')
        
        # 添加起始节点
        start_node = RRTStarNode(start_point)
        self.nodes.append(start_node)
        
        # 动态调整重布线半径
        for iteration in range(self.max_iterations):
            # 目标偏向采样
            if random.random() < self.goal_bias:
                random_point = goal_point
            else:
                random_point = self.sample_random_point()
            
            # 找到最近节点
            nearest_node = self.find_nearest_node(random_point)
            if nearest_node is None:
                continue
            
            # 扩展新点
            new_point = self.steer(nearest_node, random_point)
            
            # 碰撞检查
            if not self.is_collision_free(new_point):
                continue
            if not self.is_path_collision_free(nearest_node.position, new_point):
                continue
            
            # 找到附近节点
            current_radius = min(self.rewire_radius, 
                               math.sqrt(math.log(len(self.nodes) + 1) / len(self.nodes)) * 2.0)
            near_nodes = self.find_near_nodes(new_point, current_radius)
            
            # 选择最优父节点
            parent, cost = self.choose_parent(new_point, near_nodes)
            if parent is None:
                parent = nearest_node
                cost = parent.cost + parent.distance_to(new_point)
            
            # 创建新节点
            new_node = RRTStarNode(new_point, parent)
            new_node.cost = cost
            parent.children.append(new_node)
            self.nodes.append(new_node)
            
            # 重新连接
            self.rewire(new_node, near_nodes)
            
            # 检查是否到达目标
            if new_node.distance_to(goal_point) < goal_tolerance:
                goal_cost = new_node.cost + new_node.distance_to(goal_point)
                if goal_cost < self.best_cost:
                    self.best_cost = goal_cost
                    self.goal_node = RRTStarNode(goal_point, new_node)
                    self.goal_node.cost = goal_cost
                    print(f"找到新路径! 成本: {goal_cost:.4f}, 迭代: {iteration}")
            
            # 定期输出进度
            if iteration % 500 == 0:
                print(f"RRT*规划进度: {iteration}/{self.max_iterations}, 节点数: {len(self.nodes)}")
        
        # 生成最终路径
        if self.goal_node:
            self.best_path = self.goal_node.path_to_parent()
            print(f"RRT*规划完成! 路径长度: {len(self.best_path)}, 总成本: {self.best_cost:.4f}")
            return True
        else:
            print("RRT*规划失败: 未找到到达目标的路径")
            return False


class DualArmObstacleAvoidanceController:
    """双机械臂自主避障控制器"""
    
    def __init__(self):
        rclpy.init()
        
        # 创建节点
        self.node = Node("dual_arm_obstacle_avoidance_controller")
        
        # 创建回调组
        self.callback_group = ReentrantCallbackGroup()
        
        # 紧急停止标志
        self.emergency_stop = False
        self.planning_lock = Lock()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # 工作空间边界 (基于EA200机械臂规格)
        self.workspace_bounds = [
            (-0.895, 0.895),  # X边界
            (-0.895, 0.895),  # Y边界
            (-0.15, 1.1)      # Z边界
        ]
        
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
                "max_velocity": 0.03,  # 降低速度以提高安全性
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
        
        # 初始化路径规划器
        self.planner_left = RRTStarPlanner(
            workspace_bounds=self.workspace_bounds,
            max_iterations=1000,  # 减少迭代次数
            step_size=0.05,        # 增大步长
            goal_bias=0.2,         # 增加目标偏向
            rewire_radius=0.2      # 增加重布线半径
        )
        
        self.planner_right = RRTStarPlanner(
            workspace_bounds=self.workspace_bounds,
            max_iterations=1000,  # 减少迭代次数
            step_size=0.05,        # 增大步长
            goal_bias=0.2,         # 增加目标偏向
            rewire_radius=0.2      # 增加重布线半径
        )
        
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
        
        print("双机械臂自主避障控制器初始化完成")
    
    def signal_handler(self, signum, frame):
        """信号处理函数"""
        print("\n收到停止信号，执行紧急停止...")
        self.emergency_stop = True
        self.emergency_stop_motion()
    
    def emergency_stop_motion(self):
        """紧急停止机械臂运动"""
        print("执行紧急停止...")
        try:
            # 使用MoveIt2的停止方法 - 通过发送空的轨迹来停止
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
            
            # 停止左臂
            stop_trajectory = JointTrajectory()
            stop_trajectory.joint_names = self.arm_configs["left_arm"]["joint_names"]
            stop_point = JointTrajectoryPoint()
            stop_point.positions = []  # 空位置表示停止
            stop_trajectory.points.append(stop_point)
            
            # 停止右臂
            stop_trajectory_right = JointTrajectory()
            stop_trajectory_right.joint_names = self.arm_configs["right_arm"]["joint_names"]
            stop_trajectory_right.points.append(stop_point)
            
            print("双臂已停止")
        except Exception as e:
            print(f"紧急停止时出错: {e}")
            # 如果上述方法失败，尝试其他停止方式
            try:
                # 设置极低的速度来快速停止
                self.moveit2_left.max_velocity = 0.001
                self.moveit2_right.max_velocity = 0.001
                print("已降低速度进行紧急停止")
            except Exception as e2:
                print(f"备用停止方法也失败: {e2}")
    
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
    
    def pose_to_position_array(self, pose):
        """将位姿转换为位置数组"""
        if pose is None:
            print("警告: pose为None，返回默认位置")
            return np.array([0.3, 0.0, 0.5])
        
        try:
            return np.array([pose.position.x, pose.position.y, pose.position.z])
        except AttributeError as e:
            print(f"警告: pose对象格式错误: {e}，返回默认位置")
            return np.array([0.3, 0.0, 0.5])
    
    def setup_environment_obstacles(self):
        """设置环境障碍物（示例）"""
        print("设置环境障碍物...")
        
        # 清空现有障碍物
        self.planner_left.clear_obstacles()
        self.planner_right.clear_obstacles()
        
        # 添加一些示例障碍物（可根据实际环境调整）
        obstacles = [
            # 中央障碍物
            {'center': [0.0, 0.0, 0.5], 'radius': 0.15},
            # 左侧障碍物
            {'center': [-0.3, 0.2, 0.6], 'radius': 0.1},
            # 右侧障碍物
            {'center': [0.3, -0.2, 0.4], 'radius': 0.12},
            # 上方障碍物
            {'center': [0.0, 0.0, 0.8], 'radius': 0.08},
        ]
        
        for obstacle in obstacles:
            self.planner_left.add_obstacle(obstacle['center'], obstacle['radius'])
            self.planner_right.add_obstacle(obstacle['center'], obstacle['radius'])
        
        print(f"已添加 {len(obstacles)} 个障碍物")
    
    def get_current_pose(self, arm_name):
        """获取当前位姿"""
        if arm_name == "left_arm":
            moveit2 = self.moveit2_left
        else:
            moveit2 = self.moveit2_right
        
        # 创建默认位姿
        pose = Pose()
        pose.position.x = 0.3 if arm_name == "left_arm" else -0.3
        pose.position.y = 0.0
        pose.position.z = 0.5
        
        # 默认朝向
        w, x, y, z = self.euler_to_quaternion(0, 0, 0)
        pose.orientation.w = w
        pose.orientation.x = x
        pose.orientation.y = y
        pose.orientation.z = z
        
        try:
            # 检查关节状态是否可用
            if moveit2.joint_state is not None:
                print(f"{arm_name} 使用默认起始位姿进行规划")
            else:
                print(f"{arm_name} 关节状态不可用，使用默认位姿")
            
            return pose
            
        except Exception as e:
            print(f"获取{arm_name}当前位姿时出错: {e}，使用默认位姿")
            return pose
    
    def plan_path_to_pose(self, arm_name, target_pose):
        """使用RRT*规划到目标位姿的路径"""
        print(f"为{arm_name}规划避障路径...")
        
        # 获取规划器
        planner = self.planner_left if arm_name == "left_arm" else self.planner_right
        
        # 获取当前位置（简化处理）
        current_pose = self.get_current_pose(arm_name)
        if current_pose is None:
            print(f"无法获取{arm_name}当前位置")
            return None
        
        start_point = self.pose_to_position_array(current_pose)
        goal_point = self.pose_to_position_array(target_pose)
        
        # 检查目标点是否在障碍物内
        if not planner.is_collision_free(goal_point):
            print(f"警告: 目标位置在障碍物内，尝试寻找最近的无碰撞点")
            # 在目标点周围搜索无碰撞点
            for radius in [0.05, 0.1, 0.15, 0.2]:
                found_safe_point = False
                for _ in range(50):  # 尝试50个随机方向
                    direction = np.random.randn(3)
                    direction = direction / np.linalg.norm(direction)
                    test_point = goal_point + direction * radius
                    
                    if planner.is_collision_free(test_point):
                        goal_point = test_point
                        found_safe_point = True
                        print(f"找到替代目标点，偏移距离: {radius}m")
                        break
                
                if found_safe_point:
                    break
            else:
                print("无法找到无碰撞的目标点")
                return None
        
        # 执行RRT*规划
        with self.planning_lock:
            success = planner.plan(start_point, goal_point)
            
            if success and planner.best_path:
                print(f"{arm_name}路径规划成功，路径点数: {len(planner.best_path)}")
                return planner.best_path
            else:
                print(f"{arm_name}路径规划失败")
                return None
    
    def execute_path(self, arm_name, path, target_pose):
        """执行规划好的路径"""
        print(f"开始执行{arm_name}路径...")
        
        moveit2 = self.moveit2_left if arm_name == "left_arm" else self.moveit2_right
        
        try:
            # 逐点执行路径
            for i, waypoint in enumerate(path):
                if self.emergency_stop:
                    print("检测到紧急停止，终止路径执行")
                    return False
                
                # 创建位姿
                pose = Pose()
                pose.position.x = waypoint[0]
                pose.position.y = waypoint[1]
                pose.position.z = waypoint[2]
                
                # 使用目标姿态
                pose.orientation = target_pose.orientation
                
                print(f"{arm_name}执行路径点 {i+1}/{len(path)}: "
                      f"({waypoint[0]:.3f}, {waypoint[1]:.3f}, {waypoint[2]:.3f})")
                
                # 移动到路径点
                moveit2.move_to_pose(pose)
                
                # 等待运动完成
                if not moveit2.wait_until_executed():
                    print(f"{arm_name}路径点 {i+1} 执行失败")
                    return False
                
                # 短暂暂停以确保稳定
                time.sleep(0.1)
            
            # 最终移动到精确目标位姿
            print(f"{arm_name}移动到最终目标位姿...")
            moveit2.move_to_pose(target_pose)
            
            if moveit2.wait_until_executed():
                print(f"{arm_name}路径执行完成")
                return True
            else:
                print(f"{arm_name}最终位姿执行失败")
                return False
                
        except Exception as e:
            print(f"{arm_name}路径执行过程中出错: {e}")
            return False
    
    def move_arm_to_pose_with_obstacle_avoidance(self, arm_name, target_pose):
        """控制机械臂避障移动到目标位姿"""
        print(f"\n=== 开始{arm_name}避障运动 ===")
        print(f"目标位置: ({target_pose.position.x:.3f}, "
              f"{target_pose.position.y:.3f}, {target_pose.position.z:.3f})")
        
        # 设置环境障碍物
        self.setup_environment_obstacles()
        
        # 路径规划
        path = self.plan_path_to_pose(arm_name, target_pose)
        if path is None:
            print(f"{arm_name}路径规划失败，无法执行运动")
            return False
        
        # 路径执行
        success = self.execute_path(arm_name, path, target_pose)
        
        if success:
            print(f"{arm_name}成功到达目标位姿")
        else:
            print(f"{arm_name}运动失败")
        
        return success
    
    def move_both_arms_to_poses(self, left_target_pose, right_target_pose):
        """同时控制双臂避障运动"""
        print("\n=== 开始双臂协同避障运动 ===")
        
        # 创建线程同时执行
        left_thread = Thread(
            target=self.move_arm_to_pose_with_obstacle_avoidance,
            args=("left_arm", left_target_pose)
        )
        
        right_thread = Thread(
            target=self.move_arm_to_pose_with_obstacle_avoidance,
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


def create_target_pose(x, y, z, roll=0, pitch=0, yaw=0):
    """创建目标位姿"""
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    
    # 欧拉角转四元数
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    pose.orientation.w = w
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    
    return pose


def main():
    """主函数"""
    print("=== 双机械臂自主避障位姿控制程序 ===")
    print("使用改进的RRT*算法进行路径规划")
    print("支持Ctrl+C紧急停止")
    print()
    
    # 创建控制器
    controller = DualArmObstacleAvoidanceController()
    
    try:
        # 等待初始化完成
        time.sleep(2.0)
        
        # 示例1: 单臂避障运动
        print("\n--- 示例1: 左臂避障运动 ---")
        left_target = create_target_pose(
            x=0.4, y=0.3, z=0.6,
            roll=0, pitch=math.pi/4, yaw=0
        )
        
        success = controller.move_arm_to_pose_with_obstacle_avoidance("left_arm", left_target)
        print(f"左臂运动结果: {'成功' if success else '失败'}")
        
        time.sleep(2.0)
        
        # 示例2: 双臂协同避障运动
        print("\n--- 示例2: 双臂协同避障运动 ---")
        left_target = create_target_pose(
            x=0.35, y=0.25, z=0.5,
            roll=0, pitch=math.pi/6, yaw=math.pi/4
        )
        
        right_target = create_target_pose(
            x=-0.35, y=-0.25, z=0.5,
            roll=0, pitch=-math.pi/6, yaw=-math.pi/4
        )
        
        controller.move_both_arms_to_poses(left_target, right_target)
        
        # 示例3: 复杂避障场景
        print("\n--- 示例3: 复杂避障场景 ---")
        left_target = create_target_pose(
            x=0.5, y=-0.2, z=0.7,
            roll=math.pi/8, pitch=math.pi/3, yaw=math.pi/2
        )
        
        success = controller.move_arm_to_pose_with_obstacle_avoidance("left_arm", left_target)
        print(f"复杂场景运动结果: {'成功' if success else '失败'}")
        
        print("\n=== 所有示例执行完成 ===")
        
    except KeyboardInterrupt:
        print("\n收到键盘中断信号")
    except Exception as e:
        print(f"程序执行过程中出错: {e}")
    finally:
        # 关闭控制器
        controller.shutdown()


if __name__ == "__main__":
    main()