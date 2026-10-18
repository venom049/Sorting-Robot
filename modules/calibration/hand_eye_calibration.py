import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# ==================== 1. 参数配置 ====================
# 彩色相机内参矩阵
COLOR_INTRINSIC_MATRIX = np.array([
    [387.06130981, 0, 323.73394775],
    [0, 386.48782349, 241.54171753],
    [0, 0, 1]
], dtype=np.float64)

# 畸变系数
dist_coeffs = np.zeros((5, 1), dtype=np.float64)

# 棋盘格参数 - 您需要修改这个！
CHECKERBOARD_SIZE = (5,8)  # 最常见的尺寸：6行×9列内部角点
SQUARE_SIZE_CM = 2.5
SQUARE_SIZE_M = SQUARE_SIZE_CM / 100.0

# 图像路径
rgb_dir = "./1/"
rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")) + 
                   glob.glob(os.path.join(rgb_dir, "*.png")) + 
                   glob.glob(os.path.join(rgb_dir, "*.bmp")))
print(f"找到 {len(rgb_paths)} 张RGB图像")

# 机械臂末端位姿数据 - 支持更多数据点
#[x_cm, y_cm, z_cm, roll, pitch, yaw]
robot_poses_raw = [
    [-24.9, 30.2, -4.6, 136.7, 122.9, 86.3],
    [-15.7, 26.5, -11.1, -143.8, 113.2, 90.8],
    [-18.8, 29.2, -3.9, -120, 100.5, 107.7],
]

# 检查数据数量是否匹配
if len(robot_poses_raw) < len(rgb_paths):
    print(f"警告: 机械臂位姿数据({len(robot_poses_raw)}组)少于图像数量({len(rgb_paths)}张)")
    print("将只处理前{}张图像".format(min(len(robot_poses_raw), len(rgb_paths))))
elif len(robot_poses_raw) > len(rgb_paths):
    print(f"警告: 机械臂位姿数据({len(robot_poses_raw)}组)多于图像数量({len(rgb_paths)}张)")
    print("将只处理前{}个位姿数据".format(min(len(robot_poses_raw), len(rgb_paths))))

# ==================== 2. 图像增强函数 ====================
def enhance_checkerboard_image(img):
    """
    增强棋盘格图像，提高角点检测成功率
    返回增强后的灰度图和处理步骤的可视化
    """
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 1. CLAHE（限制对比度的自适应直方图均衡化）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    # 2. 非局部均值去噪（保留边缘）
    denoised = cv2.fastNlMeansDenoising(clahe_img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # 3. 自适应直方图均衡化（全局）
    eq_img = cv2.equalizeHist(denoised)
    # 4. 高斯模糊（轻微）
    blurred = cv2.GaussianBlur(eq_img, (3, 3), 0)
    # 5. 锐化（增强边缘）
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    # 6. 自适应阈值（备用）
    binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    # 创建处理步骤的可视化
    h, w = gray.shape
    scale = 0.5
    small_h, small_w = int(h * scale), int(w * scale)
    steps = [gray, clahe_img, denoised, eq_img, blurred, sharpened, binary]
    step_names = ['Original', 'CLAHE', 'Denoised', 'Equalized', 'Blurred', 'Sharpened', 'Binary']
    # 调整所有图像到相同大小
    steps_resized = [cv2.resize(s, (small_w, small_h)) for s in steps]
    # 创建3x3的网格显示（最后一个位置留空）
    rows = []
    for i in range(0, len(steps_resized), 3):
        row_imgs = steps_resized[i:i+3]
        # 如果不够3个，用黑色填充
        while len(row_imgs) < 3:
            row_imgs.append(np.zeros((small_h, small_w), dtype=np.uint8))
        row = np.hstack(row_imgs)
        rows.append(row)
    process_vis = np.vstack(rows)
    # 添加文字标签
    for i, name in enumerate(step_names):
        row = i // 3
        col = i % 3
        x = col * small_w + 10
        y = row * small_h + 30
        cv2.putText(process_vis, name, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return sharpened, process_vis
# ==================== 3. 鲁棒角点检测函数 ====================
def robust_checkerboard_detection(img, pattern_size, max_attempts=5):
    debug_info = {}
    # 使用增强后的图像
    enhanced_img, process_vis = enhance_checkerboard_image(img)
    # 尝试多种检测标志组合
    flag_combinations = [
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS,
        0  # 不使用特殊标志
    ]
    # 尝试多种图像版本
    image_versions = [
        ("Enhanced", enhanced_img),
        ("Original", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
        ("Binary", cv2.adaptiveThreshold(enhanced_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)),
        ("Equalized", cv2.equalizeHist(enhanced_img))
    ]
    best_corners = None
    best_method = ""
    best_image = ""
    for img_name, test_img in image_versions:
        for i, flags in enumerate(flag_combinations):
            ret, corners = cv2.findChessboardCorners(test_img, pattern_size, flags)
            if ret:
                # 亚像素精细化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(test_img, corners, (11, 11), (-1, -1), criteria)
                # 验证角点质量
                # 1. 检查角点是否大致成网格状
                corners_reshaped = corners_refined.reshape(pattern_size[0], pattern_size[1], 2)
                # 计算相邻角点距离的均匀性
                horizontal_dist = []
                vertical_dist = []
                for r in range(pattern_size[0]):
                    for c in range(pattern_size[1]-1):
                        dist = np.linalg.norm(corners_reshaped[r, c+1] - corners_reshaped[r, c])
                        horizontal_dist.append(dist)
                for r in range(pattern_size[0]-1):
                    for c in range(pattern_size[1]):
                        dist = np.linalg.norm(corners_reshaped[r+1, c] - corners_reshaped[r, c])
                        vertical_dist.append(dist)
                if horizontal_dist and vertical_dist:
                    h_std = np.std(horizontal_dist)
                    v_std = np.std(vertical_dist)
                    # 如果网格均匀性较好，接受这个结果
                    if h_std < 20 and v_std < 20:  # 阈值可以根据图像调整
                        best_corners = corners_refined
                        best_method = f"Flags_{i}"
                        best_image = img_name
                        debug_info['quality'] = {
                            'horizontal_std': h_std,
                            'vertical_std': v_std,
                            'method': best_method,
                            'image': best_image
                        }
                        return best_corners, enhanced_img, process_vis, debug_info
    # 如果没有检测到，尝试手动模式或返回None
    return None, enhanced_img, process_vis, debug_info
# ==================== 4. 辅助函数 ====================
def euler_to_rotation_matrix(roll, pitch, yaw, degrees=True):
    """将欧拉角 (ZYX顺序) 转换为旋转矩阵"""
    if degrees:
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
    
    # ZYX顺序: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R = Rz @ Ry @ Rx
    return R

def pose_to_homogeneous_matrix(pose):
    """将 [x, y, z, roll, pitch, yaw] 转换为齐次变换矩阵"""
    x_cm, y_cm, z_cm, roll, pitch, yaw = pose
    # 转换为米
    x = x_cm / 100.0
    y = y_cm / 100.0
    z = z_cm / 100.0
    
    R = euler_to_rotation_matrix(roll, pitch, yaw, degrees=True)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

def visualize_corners(img, corners, pattern_size, title="Detected Corners"):
    """可视化检测到的角点"""
    vis_img = img.copy()
    if corners is not None:
        cv2.drawChessboardCorners(vis_img, pattern_size, corners, True)
        
        # 标注角点序号
        corners_reshaped = corners.reshape(pattern_size[0], pattern_size[1], 2)
        for r in range(pattern_size[0]):
            for c in range(pattern_size[1]):
                x, y = int(corners_reshaped[r, c, 0]), int(corners_reshaped[r, c, 1])
                idx = r * pattern_size[1] + c
                cv2.putText(vis_img, str(idx), (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.circle(vis_img, (x, y), 3, (0, 0, 255), -1)
    
    # 显示图像
    cv2.imshow(title, vis_img)
    cv2.waitKey(500)  # 显示0.5秒
    
    # 同时保存图像
    cv2.imwrite(f"detected_{title.replace(' ', '_').lower()}.jpg", vis_img)
    
    return vis_img

# ==================== 5. 主程序：检测角点 ====================
print("\n" + "="*60)
print("开始鲁棒棋盘格角点检测")
print("="*60)

# 生成棋盘格的世界坐标
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_M

# 存储数据
T_cam_target_list = []
T_base_gripper_list = []
valid_indices = []
all_debug_info = []

# 确定要处理的最大数量
max_to_process = min(len(rgb_paths), len(robot_poses_raw))
print(f"将处理最多 {max_to_process} 组数据（图像和位姿配对）")

for i in range(max_to_process):
    rgb_path = rgb_paths[i]
    robot_pose = robot_poses_raw[i]
    
    print(f"\n{'='*50}")
    print(f"处理图像 {i+1}/{max_to_process}: {os.path.basename(rgb_path)}")
    print(f"机械臂位姿: {robot_pose}")
    
    # 读取图像
    img = cv2.imread(rgb_path)
    if img is None:
        print("  错误: 无法读取图像")
        continue
    
    # 鲁棒角点检测
    corners, enhanced_img, process_vis, debug_info = robust_checkerboard_detection(
        img, CHECKERBOARD_SIZE
    )
    
    if corners is None:
        print("  警告: 角点检测失败")
        
        # 显示处理步骤帮助调试
        cv2.imshow('Processing Steps', process_vis)
        cv2.waitKey(1000)
        
        # 尝试手动调整参数
        print("  尝试备选检测方法...")
        
        # 尝试不同的棋盘格尺寸（常见尺寸）
        alternative_sizes = [(5,8), (7,7), (4,11), (7,10), (9,6)]
        for alt_size in alternative_sizes:
            ret, alt_corners = cv2.findChessboardCorners(
                enhanced_img, alt_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret:
                print(f"  使用备选尺寸检测成功: {alt_size}")
                corners = alt_corners
                CHECKERBOARD_SIZE = alt_size  # 更新全局尺寸
                break
        
        if corners is None:
            print("  所有检测方法都失败，跳过此图像")
            cv2.destroyAllWindows()
            continue
    
    print(f"  成功检测到 {len(corners)} 个角点")
    
    # 可视化结果
    detected_img = visualize_corners(img, corners, CHECKERBOARD_SIZE, 
                                     f"Detected {i+1}")
    
    # 显示处理步骤
    cv2.imshow('Processing Steps', process_vis)
    cv2.waitKey(500)
    
    # 使用PnP求解相机位姿
    try:
        ret, rvec, tvec = cv2.solvePnP(objp, corners, COLOR_INTRINSIC_MATRIX, dist_coeffs)
        
        if not ret:
            print("  PnP求解失败")
            continue
        
        # 将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        
        # 构建齐次变换矩阵 T_cam_target
        T_cam_target = np.eye(4)
        T_cam_target[:3, :3] = R
        T_cam_target[:3, 3] = tvec.flatten()
        
        # 转换机械臂位姿
        T_base_gripper = pose_to_homogeneous_matrix(robot_pose)
        
        # 保存结果
        T_cam_target_list.append(T_cam_target)
        T_base_gripper_list.append(T_base_gripper)
        valid_indices.append(i)
        all_debug_info.append(debug_info)
        
        print(f"  成功计算位姿")
        print(f"  平移: {tvec.flatten()} m")
            
    except Exception as e:
        print(f"  PnP求解错误: {e}")
        continue

cv2.destroyAllWindows()

print(f"\n{'='*60}")
print(f"角点检测结果: {len(T_cam_target_list)}/{max_to_process} 组数据成功")
print("="*60)

if len(T_cam_target_list) < 2:
    print("错误: 至少需要2组成功检测的数据才能进行手眼标定")
    print("\n建议:")
    print("1. 检查 CHECKERBOARD_SIZE 是否正确")
    print("2. 确保棋盘格在图像中清晰可见")
    print("3. 调整光照条件")
    print("4. 尝试不同的棋盘格方向")
    exit()

# ==================== 6. 手眼标定 ====================
print("\n开始手眼标定...")

# 检查数据
print(f"有效数据组数: {len(T_cam_target_list)}")

# 准备数据格式
R_gripper2base = []
t_gripper2base = []
R_target2cam = []
t_target2cam = []

for i in range(len(T_cam_target_list)):
    # T_base_gripper
    T_b_g = T_base_gripper_list[i]
    R_gripper2base.append(T_b_g[:3, :3])
    t_gripper2base.append(T_b_g[:3, 3])
    
    # T_cam_target -> 需要转换为 T_target_cam
    T_c_t = T_cam_target_list[i]
    # T_target_cam = inv(T_cam_target)
    R_target2cam.append(T_c_t[:3, :3].T)  # 旋转矩阵的转置
    t_target2cam.append(-T_c_t[:3, :3].T @ T_c_t[:3, 3])  # -R^T * t

# 转换为numpy数组
R_gripper2base = np.array(R_gripper2base)
t_gripper2base = np.array(t_gripper2base)
R_target2cam = np.array(R_target2cam)
t_target2cam = np.array(t_target2cam)

print(f"数据形状:")
print(f"  R_gripper2base: {R_gripper2base.shape}")
print(f"  t_gripper2base: {t_gripper2base.shape}")
print(f"  R_target2cam: {R_target2cam.shape}")
print(f"  t_target2cam: {t_target2cam.shape}")

# 尝试OpenCV的手眼标定
try:
    print("\n使用OpenCV calibrateHandEye...")
    # 注意：对于眼在手外，我们要求解的是 T_base_cam
    R_base_cam = np.zeros((3, 3))
    t_base_cam = np.zeros(3)
    
    # 正确的调用方式
    R_base_cam, t_base_cam = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    print("OpenCV方法成功!")
    
except Exception as e:
    print(f"OpenCV方法失败: {e}")
    exit()

# 构建变换矩阵
T_base_cam = np.eye(4)
T_base_cam[:3, :3] = R_base_cam
T_base_cam[:3, 3] = t_base_cam.flatten()

print("\n=== 手眼标定结果 ===")
print(f"T_base_cam:")
print(T_base_cam)

# 验证结果
print("\n=== 验证标定结果 ===")

# 计算T_gripper_target（标定板相对于末端）
if len(T_cam_target_list) > 0:
    T_gripper_target = np.linalg.inv(T_base_gripper_list[0]) @ T_base_cam @ T_cam_target_list[0]
    
    print(f"标定板相对于末端的变换 T_gripper_target:")
    print(f"平移: {T_gripper_target[:3, 3]} m")
    
    # 验证所有数据点
    errors = []
    for i in range(len(T_cam_target_list)):
        left = T_base_cam @ T_cam_target_list[i]
        right = T_base_gripper_list[i] @ T_gripper_target
        
        # 计算误差
        position_error = np.linalg.norm(left[:3, 3] - right[:3, 3]) * 100  # cm
        
        # 旋转误差（角度）
        R_error = left[:3, :3].T @ right[:3, :3]
        angle_error = np.arccos((np.trace(R_error) - 1) / 2)
        angle_error_deg = np.degrees(angle_error)
        
        errors.append((position_error, angle_error_deg))
        
        print(f"数据{i+1}: 位置误差={position_error:.2f} cm, 角度误差={angle_error_deg:.2f}°")
    
    avg_pos_error = np.mean([e[0] for e in errors])
    avg_angle_error = np.mean([e[1] for e in errors])
    
    print(f"\n平均误差: 位置={avg_pos_error:.2f} cm, 角度={avg_angle_error:.2f}°")

# ==================== 7. 结果输出 ====================
print("\n" + "="*60)
print("手眼标定结果")
print("="*60)

print(f"\n相机相对于机器人基座的位姿 T_base_cam:")
print(f"旋转矩阵:")
print(R_base_cam)
print(f"\n平移向量 (米):")
print(t_base_cam)
print(f"\n平移向量 (厘米): {t_base_cam*100}")

# 转换为欧拉角显示
# 从旋转矩阵提取欧拉角 (ZYX顺序)
sy = np.sqrt(R_base_cam[0, 0] * R_base_cam[0, 0] + R_base_cam[1, 0] * R_base_cam[1, 0])
singular = sy < 1e-6

if not singular:
    roll = np.arctan2(R_base_cam[2, 1], R_base_cam[2, 2])
    pitch = np.arctan2(-R_base_cam[2, 0], sy)
    yaw = np.arctan2(R_base_cam[1, 0], R_base_cam[0, 0])
else:
    roll = np.arctan2(-R_base_cam[1, 2], R_base_cam[1, 1])
    pitch = np.arctan2(-R_base_cam[2, 0], sy)
    yaw = 0

print(f"\n欧拉角 (度):")
print(f"  Roll (X): {np.degrees(roll):.2f}°")
print(f"  Pitch (Y): {np.degrees(pitch):.2f}°")
print(f"  Yaw (Z): {np.degrees(yaw):.2f}°")

# ==================== 8. 验证和保存 ====================
print("\n验证标定结果:")
print("-" * 40)

# 保存结果
np.save('hand_eye_matrix.npy', T_base_cam)
print("\n✓ 手眼标定矩阵已保存为 'hand_eye_matrix.npy'")

# 保存为文本文件
with open('hand_eye_calibration_result.txt', 'w') as f:
    f.write("手眼标定结果\n")
    f.write("="*50 + "\n\n")
    f.write(f"成功数据组数: {len(T_cam_target_list)}/{max_to_process}\n\n")
    
    f.write("变换矩阵 T_base_cam:\n")
    for row in T_base_cam:
        f.write("[")
        for i, val in enumerate(row):
            f.write(f"{val:12.6f}")
            if i < 3:
                f.write(", ")
        f.write("]\n")
    
    f.write(f"\n平移 (米): {t_base_cam[0]:.6f}, {t_base_cam[1]:.6f}, {t_base_cam[2]:.6f}\n")
    f.write(f"平移 (厘米): {t_base_cam[0]*100:.2f}, {t_base_cam[1]*100:.2f}, {t_base_cam[2]*100:.2f}\n")
    f.write(f"\n欧拉角 (度):\n")
    f.write(f"  Roll (X): {np.degrees(roll):.2f}°\n")
    f.write(f"  Pitch (Y): {np.degrees(pitch):.2f}°\n")
    f.write(f"  Yaw (Z): {np.degrees(yaw):.2f}°\n")

print("详细结果已保存为 'hand_eye_calibration_result.txt'")

print("\n程序完成！")