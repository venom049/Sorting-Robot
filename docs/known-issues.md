# 已知问题与集成缺口

此处来自静态源码核对。除明确记录的语法笔误外，其余问题尚未修改，需要结合真实安装方式、机器人配置或独立测试处理。

## 已修复

`modules/motion/kinova_motion_moveit/motion_obstacle_avoidance.py` 原第 715 行包含 `)0`，导致整个文件无法解析。已删除多余的 `0`。原始代码仍保存在 ZIP 中；修改前后的哈希记录于来源清单。最终 Python 静态编译结果见 [校验报告](verification-report.md)。

## 应优先处理

| 问题 | 源码证据 | 影响与处理方向 |
| --- | --- | --- |
| 手眼变换方向不一致 | `hand_eye_calibration.py` 的 `T_c_t` 被取逆后传给 `target2cam`，返回值直接标记 `T_base_cam` | 按实际相机/标定板安装方式重建变换方程，用合成数据和独立样本验证；详见技术路线 |
| 标定样本下限错误 | 仅在有效数据少于 2 时退出 | 不应将 2 组当成充分条件；至少准备满足求解器要求的多姿态数据，并检查运动是否退化 |
| 棋盘格形状存在冲突 | 主配置 `(5,8)`，注释却写 6×9；备用检测修改尺寸后未重建 `objp` | 固定真实角点数；尺寸变化时同步生成对象点；角点重排当前使用的行列顺序也需核验 |
| 灵巧手始终跳过 | `motion_joint_dualarm_hand.py` 设置 `self.hand=None`，成功创建 publisher 后未更新；`call_hand_position()` 首先检查 `self.hand is None` | 以实际 publisher/SDK 初始化结果作为就绪条件，并增加执行确认 |
| RRT* 起点不是实测位姿 | `motion_obstacle_avoidance.py:get_current_pose()` 返回固定默认值 | 改成 TF/FK 获取真实 TCP，失败时停止任务，避免从假起点规划 |
| 避障目标可能再次回到障碍内 | `plan_path_to_pose()` 会替换碰撞目标，但 `execute_path()` 最后仍发送原始 `target_pose` | 统一最终目标并重新验证；替代目标是否允许需要由任务层决定 |
| 缺少完整 ROS2 包与机器人描述 | 归档无 `package.xml`、包级 `setup.py`、resource 标记、URDF/SRDF、launch 或 MoveIt 配置 | 需要补回原工作空间配置，才可按 `ros2 run` 方式安装和部署 |

## 文档与环境不一致

- `README_TARGET_POSE.md`、`run_target_pose.bat` 引用 `motion_target_pose.py`，归档中没有该文件。`motion_cartesian_simple.py` 有类似的交互位姿控制实现，但不能直接视作原文件的同名替代。
- `QUICK_START.md` 中出现 `motion_cartesian_pose` 命令，归档中无相应脚本及 entry point 定义；本次 README 不宣称这些命令已经可用。
- `run_with_ros2.sh`、`setup_environment.sh` 使用 `/home/rpp/rpp_ws`，且后者检查工作空间根目录 `setup.py`；需要按真实 ROS2 包布局重写部署配置。
- 目录名 Kinova 与部分 EA200 注释混用，工作空间范围是硬编码示例值。必须以实际机器人模型和关节限制为准。
- 原始 GraspNet README 记录了 PyTorch 1.6 等历史依赖，`requirements.txt` 大多未锁版本。自定义算子需要兼容的 PyTorch/CUDA/编译工具链；不能直接把所有依赖升级到最新版并假设可运行。
- GraspNet 未随仓库包含模型权重或完整训练集；示例图像并不等于训练数据集。
- 部分 `test_*.py` 为交互演示，仍导入 ROS 消息或依赖机器人环境，不能按名称视为纯离线单元测试。

## 能力边界

- GraspNet 当前输出通用抓取候选，未实现分拣类别识别或类别到箱位的映射。
- 自写 RRT* 在三维位置点与固定球体上检查碰撞，未体现连杆几何、动态场景订阅或双臂联合碰撞规划。`motion_cartesian_smart.py` 的虚拟墙是另一处 MoveIt 场景接口尝试，应单独验证。
- 平行夹爪候选尚未完成向灵巧手手形、掌部位姿和 TCP 的转换。
- 未发现统一任务状态机、实时抓取结果到运动指令的消息接口、抓取成功检测或失败恢复流程。
- 未进行仿真、真实相机或实机联调；原始文档中的“成功”“动态避障”等描述保留用于追溯，不作为本次验证结论。

原始文件与来源提交保留于来源清单，便于通过 Git 历史追溯本次整理。
