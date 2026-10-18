#!/usr/bin/env python3
"""
双机械臂避障控制运行脚本
简化的用户接口，支持命令行参数配置目标位姿
"""

import sys
import math
from motion_obstacle_avoidance import DualArmObstacleAvoidanceController, create_target_pose


def print_usage():
    """打印使用说明"""
    print("使用方法:")
    print("python run_obstacle_avoidance.py [选项]")
    print()
    print("选项:")
    print("  --demo                    运行预设演示场景")
    print("  --left x y z r p y        控制左臂到指定位姿")
    print("  --right x y z r p y       控制右臂到指定位姿")
    print("  --both lx ly lz lr lp ly rx ry rz rr rp ry  双臂协同控制")
    print()
    print("参数说明:")
    print("  x, y, z: 位置坐标(米)")
    print("  r, p, y: 欧拉角(弧度) - roll, pitch, yaw")
    print()
    print("示例:")
    print("  python run_obstacle_avoidance.py --demo")
    print("  python run_obstacle_avoidance.py --left 0.4 0.3 0.6 0 0.785 0")
    print("  python run_obstacle_avoidance.py --both 0.3 0.2 0.5 0 0.5 0.5 -0.3 -0.2 0.5 0 -0.5 -0.5")


def run_demo(controller):
    """运行演示场景"""
    print("=== 运行演示场景 ===")
    
    # 场景1: 左臂避障运动
    print("\n场景1: 左臂避障运动到前方位置")
    left_target = create_target_pose(0.4, 0.2, 0.6, 0, math.pi/4, 0)
    success = controller.move_arm_to_pose_with_obstacle_avoidance("left_arm", left_target)
    print(f"场景1结果: {'成功' if success else '失败'}")
    
    if not success:
        return
    
    # 场景2: 右臂避障运动
    print("\n场景2: 右臂避障运动到侧方位置")
    right_target = create_target_pose(-0.35, -0.25, 0.5, 0, -math.pi/6, -math.pi/4)
    success = controller.move_arm_to_pose_with_obstacle_avoidance("right_arm", right_target)
    print(f"场景2结果: {'成功' if success else '失败'}")
    
    if not success:
        return
    
    # 场景3: 双臂协同运动
    print("\n场景3: 双臂协同避障运动")
    left_target = create_target_pose(0.3, 0.3, 0.4, 0, math.pi/6, math.pi/4)
    right_target = create_target_pose(-0.3, -0.3, 0.4, 0, -math.pi/6, -math.pi/4)
    controller.move_both_arms_to_poses(left_target, right_target)


def parse_pose_args(args, start_idx):
    """解析位姿参数"""
    if len(args) < start_idx + 6:
        print("错误: 位姿参数不足，需要6个参数 (x y z roll pitch yaw)")
        return None
    
    try:
        x = float(args[start_idx])
        y = float(args[start_idx + 1])
        z = float(args[start_idx + 2])
        roll = float(args[start_idx + 3])
        pitch = float(args[start_idx + 4])
        yaw = float(args[start_idx + 5])
        
        return create_target_pose(x, y, z, roll, pitch, yaw)
    except ValueError:
        print("错误: 位姿参数格式不正确")
        return None


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print_usage()
        return
    
    # 创建控制器
    print("初始化双机械臂避障控制器...")
    controller = DualArmObstacleAvoidanceController()
    
    try:
        # 等待初始化完成
        import time
        time.sleep(2.0)
        
        args = sys.argv[1:]
        
        if args[0] == "--demo":
            run_demo(controller)
        
        elif args[0] == "--left":
            if len(args) < 7:
                print("错误: --left 选项需要6个位姿参数")
                print_usage()
                return
            
            target_pose = parse_pose_args(args, 1)
            if target_pose:
                print(f"控制左臂运动到指定位姿")
                success = controller.move_arm_to_pose_with_obstacle_avoidance("left_arm", target_pose)
                print(f"运动结果: {'成功' if success else '失败'}")
        
        elif args[0] == "--right":
            if len(args) < 7:
                print("错误: --right 选项需要6个位姿参数")
                print_usage()
                return
            
            target_pose = parse_pose_args(args, 1)
            if target_pose:
                print(f"控制右臂运动到指定位姿")
                success = controller.move_arm_to_pose_with_obstacle_avoidance("right_arm", target_pose)
                print(f"运动结果: {'成功' if success else '失败'}")
        
        elif args[0] == "--both":
            if len(args) < 13:
                print("错误: --both 选项需要12个位姿参数")
                print("左臂6个参数 + 右臂6个参数")
                print_usage()
                return
            
            left_target = parse_pose_args(args, 1)
            right_target = parse_pose_args(args, 7)
            
            if left_target and right_target:
                print(f"控制双臂协同运动")
                controller.move_both_arms_to_poses(left_target, right_target)
        
        else:
            print(f"错误: 未知选项 {args[0]}")
            print_usage()
    
    except KeyboardInterrupt:
        print("\n收到键盘中断信号")
    except Exception as e:
        print(f"程序执行过程中出错: {e}")
    finally:
        # 关闭控制器
        controller.shutdown()


if __name__ == "__main__":
    main()