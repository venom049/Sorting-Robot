# 按模块阅读与运行

以下路径均相对于仓库根目录。命令是按现有源码整理的使用方法，运行完整模型或机器人控制仍依赖外部环境；本次实际执行的仅是静态检查，详见 [verification-report.md](verification-report.md)。

## 1. 无硬件静态检查

```bash
python3 tools/verify_repository.py
```

只需 Python 3 和可选 Bash。检查语法、ZIP CRC/SHA-256、迁移文件哈希及项目文档本地链接。不会导入模型、初始化 ROS2 或发送运动指令；历史 `test_*.py` 不会被执行。

## 2. GraspNet 离线推理

先阅读 [上游 README](../third_party/graspnet-baseline/README.md)。其依赖说明为历史基线，需准备经过验证的 Python、PyTorch、CUDA 与编译器组合；本次未生成虚假的锁定依赖文件。

安装入口（在准备好的独立感知环境中执行）：

```bash
cd third_party/graspnet-baseline
python3 -m pip install -r requirements.txt
(cd pointnet2 && python3 setup.py install)
(cd knn && python3 setup.py install)
```

此外需要按上游 README 安装 GraspNet API，并自行准备兼容的预训练 checkpoint。`pointnet2/` 与 `knn/` 有 CUDA/C++ 扩展，代码中的 CPU 分支不代表整个网络已支持无 CUDA 环境运行。

在该模块目录中，使用已下载权重的实际路径：

```bash
python3 demo.py --checkpoint_path /absolute/path/to/checkpoint.tar
```

默认数据来自随仓库保留的 `doc/example_data/`，输出为 Open3D 抓取可视化窗口。原始 `command_demo.sh` 指向不存在于仓库的 `logs/log_kn/checkpoint.tar`，应修改为自己的权重路径或直接用上述参数。

接自有数据需要修改 `demo.py:get_and_process_data()`，匹配图像尺寸、相机内参、RGB/深度对齐方式与 `factor_depth`。当前主函数固定读取 `doc/example_data`，没有数据目录 CLI 参数。空有效点云也需要在实时集成时增加保护。

训练与评估入口为 `train.py`、`test.py`；数据集、容差标签与 checkpoint 参数按上游说明配置。

## 3. 手眼标定

依赖：NumPy、OpenCV、Matplotlib；脚本会创建图形窗口。原始图像和真实位姿没有随仓库提供。

使用前按 [已知问题](known-issues.md) 核验/修正标定方向、棋盘格参数与数据配对。建议把实验数据和输出放入独立目录；从该目录启动脚本，使相对路径 `./1/` 指向实验图像：

```text
calibration-session/
└── 1/
    ├── 001.png
    ├── 002.png
    └── ...
```

```bash
python3 /absolute/path/to/Sorting-Robot/modules/calibration/hand_eye_calibration.py
```

当前脚本仍需直接编辑内参、畸变系数、棋盘格参数和 `robot_poses_raw`。位姿输入平移为厘米、角度为度；内部转换为米。图像通过字典序排序与位姿顺序配对，文件名应采用补零编号。输出写入启动时的当前目录：角点可视化图片、`hand_eye_matrix.npy`、`hand_eye_calibration_result.txt`。

缺少图像时程序会报告数据不足并退出；不能把这个退出当作标定成功。

## 4. ROS2 / MoveIt2 运动模块

本模块源自已有工作空间中的脚本目录，**尚不是可独立 colcon 构建的完整包**。原有 `ros2 run kinova_motion_moveit ...` 示例依赖未随归档提供的包注册和 entry points。

前置条件：实际 ROS2 环境、机器人驱动与描述、MoveIt2 配置、pymoveit2、相关消息/服务包；灵巧手模块另需 Linker Hand ROS2 SDK 与实际手部接口。旧文档以 Humble 为例，没有据此验证更高版本兼容性。

先核对规划组、基座、TCP、关节名和实际驱动。准备好环境、在仿真中验证后，脚本路径入口为：

```bash
cd modules/motion/kinova_motion_moveit
python3 motion_joint_dualarm.py
```

其他入口：

| 目的 | 文件 | 单位/前提 |
| --- | --- | --- |
| 交互笛卡尔目标 | `motion_cartesian_simple.py` | 位置米；用户输入欧拉角度 |
| IK 与关节连续性 | `motion_cartesian_smart.py` | 需要 `/compute_ik` 与对应 MoveIt 配置 |
| 关节轨迹示例 | `motion_trajectory_dualarm.py` | 运行前检查内部预设轨迹点 |
| RRT* 示例 | `run_obstacle_avoidance.py` | CLI 欧拉角为弧度；当前存在已记录的规划逻辑问题 |
| 双臂与手部联动 | `motion_joint_dualarm_hand.py` | 手部就绪门控需要修复 |

这些是运动执行程序，部分会按预设目标循环运行。不能把 `test_obstacle_avoidance.py` 的名称当作只做静态检查的依据。当前阶段应先补齐配置并解决已知逻辑问题，再进入机器人联调。

原始 `.sh`/`.bat` 和旧文档为追溯保留；其中机器路径与缺失入口没有改成未经验证的新启动命令。
