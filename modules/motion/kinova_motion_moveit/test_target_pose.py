#!/usr/bin/env python3
"""
目标位姿控制程序测试版本
不依赖ROS2实际连接，用于验证程序逻辑和输入处理
"""

import time
import math
from geometry_msgs.msg import Pose


class TargetPoseTester:
    """目标位姿测试类"""
    
    def __init__(self):
        print("=== 目标位姿控制程序测试版本 ===")
        print("此版本不连接实际机械臂，用于验证程序逻辑")
        print()
    
    def check_workspace_limits(self, x, y, z):
        """
        检查位姿是否在EA200机械臂工作空间内
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
    
    def euler_to_quaternion(self, thetax_deg, thetay_deg, thetaz_deg):
        """将欧拉角转换为四元数"""
        # 将角度转换为弧度
        thetax_rad = math.radians(thetax_deg)
        thetay_rad = math.radians(thetay_deg)
        thetaz_rad = math.radians(thetaz_deg)
        
        # 计算半角
        cy = math.cos(thetaz_rad * 0.5)
        sy = math.sin(thetaz_rad * 0.5)
        cp = math.cos(thetay_rad * 0.5)
        sp = math.sin(thetay_rad * 0.5)
        cr = math.cos(thetax_rad * 0.5)
        sr = math.sin(thetax_rad * 0.5)
        
        # 计算四元数分量
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return [x, y, z, w]
    
    def create_target_pose(self, x, y, z, thetax_deg, thetay_deg, thetaz_deg):
        """从位置和欧拉角创建位姿对象"""
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        
        # 将欧拉角转换为四元数
        quaternion = self.euler_to_quaternion(thetax_deg, thetay_deg, thetaz_deg)
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]
        
        return pose
    
    def simulate_motion(self, arm_name, target_pose):
        """模拟机械臂运动"""
        print(f"\n=== 模拟{arm_name}运动 ===")
        print(f"目标位置: ({target_pose.position.x:.3f}, {target_pose.position.y:.3f}, {target_pose.position.z:.3f}) m")
        print(f"目标姿态: ({target_pose.orientation.x:.4f}, {target_pose.orientation.y:.4f}, {target_pose.orientation.z:.4f}, {target_pose.orientation.w:.4f})")
        
        # 模拟运动过程
        print("开始运动...")
        for i in range(5):
            time.sleep(0.5)
            print(f"  运动进度: {(i+1)*20}%")
        
        print("✓ 模拟运动完成")
        return True
    
    def select_arm(self):
        """选择要控制的机械臂"""
        print("请选择要控制的机械臂:")
        print("1. 左臂")
        print("2. 右臂")
        print("3. 双臂同时")
        
        while True:
            try:
                choice = input("请输入选择 (1, 2 或 3): ").strip()
                if choice == "1":
                    return ["left_arm"], ["左臂"]
                elif choice == "2":
                    return ["right_arm"], ["右臂"]
                elif choice == "3":
                    return ["left_arm", "right_arm"], ["左臂", "右臂"]
                else:
                    print("无效选择，请输入 1, 2 或 3")
            except KeyboardInterrupt:
                print("\n程序被用户中断")
                return None, None
    
    def input_target_pose(self, arm_display_name=""):
        """输入目标位姿坐标"""
        print(f"\n请输入{arm_display_name}的目标位姿坐标:")
        print("坐标系说明：")
        print("  - 位置：相对于机械臂基座坐标系（米）")
        print("  - 姿态：使用欧拉角（度）")
        print("    • thetax: 绕X轴旋转角度")
        print("    • thetay: 绕Y轴旋转角度") 
        print("    • thetaz: 绕Z轴旋转角度")
        print("    • 旋转顺序：ZYX（先绕Z轴，再绕Y轴，最后绕X轴）")
        
        while True:
            try:
                print(f"\n--- {arm_display_name}目标位姿参数 ---")
                x = input("位置 X (米): ").strip()
                y = input("位置 Y (米): ").strip()
                z = input("位置 Z (米): ").strip()
                thetax = input("姿态 thetax (绕X轴旋转角度，度): ").strip()
                thetay = input("姿态 thetay (绕Y轴旋转角度，度): ").strip()
                thetaz = input("姿态 thetaz (绕Z轴旋转角度，度): ").strip()
                
                # 验证输入
                if not all([x, y, z, thetax, thetay, thetaz]):
                    print("所有参数都必须输入")
                    continue
                
                # 转换为浮点数
                x, y, z = float(x), float(y), float(z)
                thetax, thetay, thetaz = float(thetax), float(thetay), float(thetaz)
                
                # 检查工作空间限制
                is_valid, message = self.check_workspace_limits(x, y, z)
                if not is_valid:
                    print(f"\n⚠️  警告: {message}")
                    print("建议调整位置参数后重新输入")
                    continue
                
                # 创建位姿对象
                pose = self.create_target_pose(x, y, z, thetax, thetay, thetaz)
                
                # 显示转换后的四元数（供调试参考）
                print(f"\n{arm_display_name}目标位姿参数:")
                print(f"  位置: x={x:.3f}m, y={y:.3f}m, z={z:.3f}m")
                print(f"  姿态: thetax={thetax}°, thetay={thetay}°, thetaz={thetaz}°")
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
    
    def execute_test(self):
        """执行测试"""
        print("开始目标位姿控制测试...")
        
        # 选择要控制的机械臂
        arm_names, arm_display_names = self.select_arm()
        if not arm_names:
            return False
        
        print(f"\n控制对象: {', '.join(arm_display_names)}")
        
        # 输入目标位姿
        target_poses = []
        for arm_display_name in arm_display_names:
            if len(arm_display_names) == 1:
                # 单臂控制
                target_pose = self.input_target_pose(arm_display_name)
                if not target_pose:
                    print("无法获取目标位姿，程序退出")
                    return False
                target_poses.append(target_pose)
            else:
                # 双臂控制
                target_pose = self.input_target_pose(arm_display_name)
                if not target_pose:
                    print(f"无法获取{arm_display_name}目标位姿，程序退出")
                    return False
                target_poses.append(target_pose)
        
        # 模拟运动
        print("\n=== 开始模拟运动 ===")
        for i, (arm_display_name, target_pose) in enumerate(zip(arm_display_names, target_poses)):
            success = self.simulate_motion(arm_display_name, target_pose)
            if not success:
                print(f"{arm_display_name}模拟运动失败")
                return False
            
            # 双臂运动时暂停一下
            if len(arm_display_names) > 1 and i == 0:
                time.sleep(1)
        
        print(f"\n=== 测试完成 ===")
        print("所有参数输入正确，工作空间检查通过")
        print("如需连接实际机械臂，请使用 motion_target_pose.py")
        return True


def main():
    """主函数"""
    print("目标位姿控制程序 - 测试版本")
    print("此版本用于验证输入参数和程序逻辑，不连接实际机械臂")
    print("适用于ROS2环境未完全配置时的测试")
    print()
    
    tester = TargetPoseTester()
    
    try:
        while True:
            success = tester.execute_test()
            
            if success:
                print("\n测试成功！")
                
                # 询问是否继续测试
                choice = input("\n是否继续测试？(y/n): ").strip().lower()
                if choice != 'y' and choice != 'yes':
                    break
            else:
                print("\n测试失败")
                break
                
        print("\n感谢使用目标位姿控制程序测试版本！")
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行过程中出错: {e}")


if __name__ == "__main__":
    main()