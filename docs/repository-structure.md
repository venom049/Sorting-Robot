# 仓库结构与迁移记录

## 整理原则

按职责划分：自有模块放入 `modules/`，外部算法放入 `third_party/`，项目说明放入 `docs/`，原始归档与历史材料放入 `archives/`。保留 GraspNet 内部目录、原始许可和相对路径；保留运动模块内部的兄弟模块导入关系与原始文档，避免在本次整理中扩大运行逻辑改动。

根目录原先只有 README、两个 ZIP、一个标定脚本及空占位文件。现在可直接浏览解压代码，根目录 README 提供统一阅读入口。

## 路径映射

| 原始路径/归档成员 | 整理后路径 | 处理方式 |
| --- | --- | --- |
| `README.md` | `README.md`；原文另存 `archives/legacy/README.original.md` | 重写项目入口并保留原文 |
| `graspnet-baseline-main.zip` | `archives/graspnet-baseline-main.zip` | 原包逐字节保留 |
| ZIP 内 `graspnet-baseline-main/` | `third_party/graspnet-baseline/` | 完整解压，文件内容不变 |
| `kinova_motion_moveit.zip` | `archives/kinova_motion_moveit.zip` | 原包逐字节保留 |
| ZIP 内 `kinova_motion_moveit/` | `modules/motion/kinova_motion_moveit/` | 解压源代码及文档 |
| `hand-eye calibration.py` | `modules/calibration/hand_eye_calibration.py` | 规范文件名，内容不变 |
| `graspnet_inference` | `archives/legacy/graspnet_inference` | 原样保留，仅含换行，无推理逻辑 |
| `motion_cartesian_simple（复件）.py` | `archives/legacy/motion_cartesian_simple（复件）.py` | 历史版本归档，和当前版本内容不同，未删除或合并 |
| ZIP 内 `build/`、`__pycache__/` | 保留在原 ZIP 中，不保留解压副本 | 构建标记和平台相关字节码不进入源码目录 |

唯一源代码内容修正：`motion_obstacle_avoidance.py` 原第 715 行的 `)0` 改为 `)`，消除 Python 语法错误。算法、运动目标、坐标系、硬件参数与手部动作逻辑未作功能改写。

解压成员的逐文件来源、原始 SHA-256、整理后 SHA-256 及去向见 [source-manifest.json](source-manifest.json)。其中被排除的文件仍可从原始 ZIP 恢复。

## 模块职责与使用路径

- `modules/calibration/`：手眼标定工具；脚本依赖当前工作目录中的 `1/` 图像目录。
- `modules/motion/kinova_motion_moveit/`：保留原始 ROS2 控制脚本和说明。它目前是源码目录，不能因目录名而视作完整可构建的 ROS2 包。
- `third_party/graspnet-baseline/`：感知模型、数据处理、训练、评估、示例图像和 CUDA/C++ 算子。从此目录启动原有相对路径命令。
- `archives/legacy/`：追溯材料，不作为默认运行入口。
- `tools/verify_repository.py`：完整性、语法与项目文档本地链接校验，无硬件调用。

运行目录变化的影响：GraspNet 示例使用相对路径 `doc/example_data`，应先进入其模块目录；标定脚本使用 `./1/` 和当前目录输出，建议在独立实验目录运行；运动脚本中原有 `/home/rpp/rpp_ws` 为作者机器路径，新布局不会自动修改这些部署路径。详见 [运行说明](getting-started.md)。

## 后续结构扩展

实际开始实现后再增加相机节点、推理接口、配置文件和任务调度模块，避免创建名称齐全但内容为空的“完整系统”。机器人描述、MoveIt 配置、相机内参、TCP 和标定结果应版本化管理；模型权重、原始实验图像、构建产物和日志应使用独立存储或忽略规则。

当前保留原始 ZIP 用于交付可追溯。团队确认解压代码后，可另行决定是否从日常开发分支移除重复 ZIP 或迁移到 release 存储；本次未删除这些原始包。
