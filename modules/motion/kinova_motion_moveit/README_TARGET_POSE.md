# 目标位姿控制程序使用指南

## 程序说明
`motion_target_pose.py` 是一个直接输入末端执行器目标位姿(x,y,z,thetax,thetay,thetaz)来控制机械臂运动的程序。

## 环境要求
1. **ROS2 Humble** (或更高版本)
2. **Python 3.8+**
3. **必要依赖包**:
   - `rclpy`
   - `pymoveit2`
   - `numpy`
   - `geometry_msgs`

## Windows环境下运行步骤

### 1. 环境配置
```cmd
# 1. 打开命令提示符或PowerShell
# 2. 配置ROS2环境
call C:\opt\ros\humble\setup.bat

# 3. 启动ROS2守护进程
ros2 daemon start

# 4. 运行环境配置脚本
setup_windows_environment.bat
```

### 2. 启动机械臂节点
在另一个终端中启动机械臂控制节点（具体命令取决于您的机器人配置）：
```cmd
# 示例命令（需要根据实际机器人调整）
call C:\opt\ros\humble\setup.bat
# 启动机器人驱动节点
```

### 3. 运行目标位姿控制程序
```cmd
# 方法1：使用启动脚本
run_target_pose.bat

# 方法2：直接运行
python motion_target_pose.py
```

## 程序使用说明

### 1. 选择机械臂
程序启动后会提示选择要控制的机械臂：
- 1: 左臂
- 2: 右臂
- 3: 双臂同时

### 2. 输入目标位姿参数
需要输入6个参数：
- **位置坐标**（米）：
  - X: 前后方向
  - Y: 左右方向
  - Z: 上下方向
- **姿态角度**（度）：
  - thetax: 绕X轴旋转角度
  - thetay: 绕Y轴旋转角度
  - thetaz: 绕Z轴旋转角度

### 3. 坐标系说明
- **坐标系**：相对于机械臂基座坐标系
- **旋转顺序**：ZYX（先绕Z轴，再绕Y轴，最后绕X轴）
- **工作空间范围**：
  - 水平距离：最大0.895米
  - 垂直范围：-0.15米 到 1.1米

## 常见问题解决

### 问题1：关节状态超时
**现象**：程序提示"等待关节状态超时"

**解决方案**：
1. 检查ROS2是否正常运行：
   ```cmd
   ros2 node list
   ros2 topic list | findstr joint_states
   ```

2. 确保机械臂驱动节点已启动：
   ```cmd
   # 查看当前运行的节点
   ros2 node list
   # 应该能看到类似 robot_state_publisher、move_group 等节点
   ```

3. 检查关节状态话题：
   ```cmd
   ros2 topic echo /joint_states
   ```

### 问题2：工作空间超出范围
**现象**：程序提示"水平距离超出最大伸展距离"等警告

**解决方案**：
1. 检查输入的目标位置是否在合理范围内
2. 参考以下安全位置范围：
   - X: -0.5 到 0.5 米
   - Y: -0.5 到 0.5 米  
   - Z: 0.1 到 0.8 米

### 问题3：四元数转换错误
**现象**：程序运行但机械臂不运动或运动异常

**解决方案**：
1. 检查欧拉角输入是否在合理范围内（建议-180°到180°）
2. 尝试简单的姿态值（如0,0,0）进行测试

## 示例使用场景

### 示例1：左臂移动到简单位置
```
选择机械臂: 1 (左臂)
输入参数:
  X: 0.3
  Y: 0.2
  Z: 0.4
  thetax: 0
  thetay: 0
  thetaz: 0
```

### 示例2：右臂带姿态的移动
```
选择机械臂: 2 (右臂)
输入参数:
  X: -0.3
  Y: -0.2
  Z: 0.5
  thetax: 45
  thetay: 30
  thetaz: 0
```

### 示例3：双臂协同
```
选择机械臂: 3 (双臂同时)
左臂参数:
  X: 0.3, Y: 0.2, Z: 0.4
  thetax: 0, thetay: 0, thetaz: 0
右臂参数:
  X: -0.3, Y: -0.2, Z: 0.4
  thetax: 0, thetay: 0, thetaz: 0
```

## 调试技巧

1. **查看ROS2话题**：
   ```cmd
   ros2 topic list
   ros2 topic echo /joint_states
   ros2 topic echo /tf
   ```

2. **查看节点图**：
   ```cmd
   ros2 run rqt_graph rqt_graph
   ```

3. **检查MoveIt规划器**：
   ```cmd
   ros2 service list | grep moveit
   ```

## 安全注意事项

1. **确保工作空间内无障碍物**
2. **缓慢移动，避免碰撞**
3. **准备急停按钮**
4. **在测试环境中先验证程序**

## 技术支持

如遇到问题，请检查：
1. ROS2环境是否正确配置
2. 机械臂硬件连接是否正常
3. 所有依赖包是否正确安装
4. 网络连接是否稳定（如使用远程ROS2）