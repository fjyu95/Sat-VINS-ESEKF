import math
import os
import pickle
import sys
import subprocess
import json
import functools
from pathlib import Path

import cv2
import exiftool
import numpy as np
from datetime import datetime

import pandas as pd
import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown

from utils.parser import get_all_gps_gt_enu


class Logger:
    def __init__(self, log_dir="logs", also_print=True):
        self.log_dir = log_dir
        self.also_print = also_print
        self.terminal = sys.stdout
        self.log = None

    def write(self, message):
        try:
            # 确保日志目录存在
            os.makedirs(self.log_dir, exist_ok=True)

            # 按时间生成日志文件
            if not self.log:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(self.log_dir, f"script_log_{timestamp}.log")
                self.log = open(log_file, 'w')
                print(f'logfile: {log_file}')

            # 同时输出到控制台和文件
            if self.also_print:
                self.terminal.write(message)
            self.log.write(message)
            self.log.flush()  # 立即写入文件
        except Exception as e:
            self.terminal.write(f"\n[Logger Error] Failed to write log: {e}\n")
            self.terminal.flush()
            sys.exit(1)

    def flush(self):
        if self.also_print:
            self.terminal.flush()
        if self.log:
            self.log.flush()


def timer(func):
    # print(f'{timer.__name__} is a func decorator.')

    @functools.wraps(func)  # 保留func函数元信息
    def warpper(*args, **kwargs):
        start = datetime.now()
        print(f'start {func.__name__} {start} ')
        # print(func.__doc__)  # 接口文档信息
        ret = func(*args, **kwargs)
        end = datetime.now()
        print(f'end {func.__name__} {end} cost {end - start}\n')
        return ret

    return warpper


def get_params_num(model):
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    print("可训练参数量：", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # print(model) # 简单查看模型结构

    print(model.forward.__doc__)
    # summary(model, input_size=(1, 3, 224, 224))

    return


def verify_gpu():
    # 获取 CUDA_VISIBLE_DEVICES 环境变量
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')  # 环境变量中不存在则返回空字符串
    visible_devices = [int(dev) for dev in cuda_visible_devices.split(',') if dev]

    if torch.cuda.is_available():
        # 获取当前使用的逻辑 GPU 编号
        current_device = torch.cuda.current_device()
        if visible_devices:
            # 映射到物理 GPU 编号
            physical_device = visible_devices[current_device]
            print(f"当前使用的物理 GPU 编号是: {physical_device}")
        else:
            print(f"当前使用的逻辑 GPU 编号是: {current_device}")
    else:
        print("未找到可用的 GPU")


def get_gpu_memory_pynvml():
    """
    获取 GPU 的真实可用显存，使用 pynvml（NVIDIA Management Library）,适用于监控整个系统的 GPU 资源
    """
    nvmlInit()  # 初始化 NVIDIA 管理库
    print(f"{'——' * 20} nvml {'——' * 20}")

    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    for i in range(torch.cuda.device_count()):
        physical_id = int(visible_gpus[i]) if visible_gpus[0] else i
        handle = nvmlDeviceGetHandleByIndex(physical_id)
        mem_info = nvmlDeviceGetMemoryInfo(handle)

        print(f"GPU {physical_id}: {torch.cuda.get_device_name(i)}")
        print(f"  Total Memory: {mem_info.total / (1024 ** 2):.2f} MB")
        print(f"  Used Memory: {mem_info.used / (1024 ** 2):.2f} MB")
        print(f"  Free Memory: {mem_info.free / (1024 ** 2):.2f} MB\n")

    nvmlShutdown()  # 释放 NVML 资源


def get_gpu_memory_nvidia_smi():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        text=True
    )
    print(f"{'——' * 20} nvidia-smi {'——' * 20}")
    for i, line in enumerate(result.stdout.strip().split("\n")):
        total, free, used = map(int, line.split(", "))
        print(f"GPU {i}:")
        print(f"  Total Memory: {total} MB")
        print(f"  Used Memory: {used} MB")
        print(f"  Free Memory: {free} MB\n")


def get_most_free_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE,
        text=True
    )
    free_mem = [int(x) for x in result.stdout.strip().split("\n")]
    best_gpu = int(max(range(len(free_mem)), key=lambda i: free_mem[i]))  # 找到剩余显存最多的 GPU
    return best_gpu


def format_size(bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} TB"


def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def timestamp_gen(date_str, time_str):
    # date_str = "2024.10.12"
    # time_str = "10:24:04"

    datetime_str = f'{date_str} {time_str}'
    dt = datetime.strptime(datetime_str, "%Y.%m.%d %H:%M:%S:%f")
    timestamp = dt.timestamp()

    return timestamp


def extrinsic_to_c2w(rvec, tvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    xyz = -rotation_matrix.T @ tvec.reshape(-1)
    return xyz


def save_images(filename, img, save_dir='output'):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, img)
    print(f'image saved to {save_path}')


def show_images(folder='output'):
    def scale_if_wide(img, max_w=1080):
        h, w = img.shape[:2]
        if w > max_w:
            # 等比例缩放
            new_h = int(h * max_w / w)
            img = cv2.resize(img, (max_w, new_h), interpolation=cv2.INTER_AREA)
        return img

    if not folder:
        exit(f"{folder} not specified,exiting...")

    # 2. 读取并排序
    image_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    files = [f for f in os.listdir(folder) if f.lower().endswith(image_exts)]

    if not files:
        exit("no image files found,exiting...")

    img_paths = [os.path.join(folder, f) for f in files]
    img_paths.sort(key=os.path.getmtime, reverse=True)  # 按修改时间排序（新到旧）

    idx = 0
    win_name = "image viewer"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1080, 720)

    while True:
        path = img_paths[idx]
        img = cv2.imread(path)
        if img is None:
            continue

        scaled = scale_if_wide(img)
        # win_name = path
        cv2.imshow(win_name, scaled)
        print(f"Showing {path} ({idx + 1}/{len(img_paths)})")

        key = cv2.waitKey(0) & 0xFF
        if key == 27 or key == ord('q'):  # ESC / q
            break
        elif key == ord('s') or key == ord('d') or key == 83:  # → / n
            idx = (idx + 1) % len(img_paths)
        elif key == ord('w') or key == ord('a') or key == 81:  # ← / p
            idx = (idx - 1) % len(img_paths)
    cv2.destroyAllWindows()


def rotate_point(x, y, angle_deg, W, H):
    """旋转图像时，对应点坐标变换"""
    if angle_deg == 0:
        return x, y
    elif angle_deg == 90:
        return H - 1 - y, x
    elif angle_deg == 180:
        return W - 1 - x, H - 1 - y
    elif angle_deg == 270:
        return y, W - 1 - x
    else:
        raise ValueError("仅支持角度 0, 90, 180, 270")


def rotate_points(points: np.ndarray, angle_deg: int, H: int, W: int) -> np.ndarray:
    """
    旋转图像时，对应 N×2 点坐标变换
    - points: shape=(N, 2)，每行为一个 (x, y) 点
    - angle_deg: 旋转角度，仅支持 0, 90, 180, 270
    - W, H: 原图的宽度和高度
    """
    points = np.asarray(points)
    x, y = points[:, 0], points[:, 1]

    if angle_deg == 0:
        x_new, y_new = x, y
    elif angle_deg == 90:
        x_new, y_new = H - 1 - y, x
    elif angle_deg == 180:
        x_new, y_new = W - 1 - x, H - 1 - y
    elif angle_deg == 270:
        x_new, y_new = y, W - 1 - x
    else:
        raise ValueError("仅支持角度 0, 90, 180, 270")

    return np.stack([x_new, y_new], axis=1)


def rotate_by_90_multiple(img, angle_quant):
    """
    仅支持 angle_quant ∈ {0,90,180,270}，用转置/翻转实现不插值的旋转。
    返回旋转后的图（内容完整）。
    """
    if angle_quant % 360 == 0:
        return img.copy()
    elif angle_quant % 360 == 90:
        # 90° 顺时针：转置然后左右翻转
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle_quant % 360 == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle_quant % 360 == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("angle_quant must be multiple of 90")


def crop_and_expand(image, x_min, x_max, y_min, y_max, expand_ratio=0.1):
    """
    裁剪图像指定区域并向四周扩展固定比例

    参数:
        image: 输入图像（BGR或灰度图）
        x_min, x_max: 原始裁剪区域的x范围（左、右）
        y_min, y_max: 原始裁剪区域的y范围（上、下）
        expand_ratio: 扩展比例（相对于原始裁剪区域的宽高），默认0.1（10%）

    返回:
        expanded_img: 扩展后的图像
        (new_x_min, new_x_max, new_y_min, new_y_max): 扩展后的坐标范围（相对于原图）
    """
    # 获取图像尺寸
    h, w = image.shape[:2]

    # 计算原始裁剪区域的宽和高
    crop_width = x_max - x_min
    crop_height = y_max - y_min

    # 计算扩展像素（基于原始裁剪区域的宽高）
    expand_w = int(crop_width * expand_ratio)
    expand_h = int(crop_height * expand_ratio)

    # 扩展区域
    x_min_exp = max(0, x_min - expand_w)
    x_max_exp = min(w, x_max + expand_w)
    y_min_exp = max(0, y_min - expand_h)
    y_max_exp = min(h, y_max + expand_h)

    # 裁剪图像
    cropped = image[y_min_exp:y_max_exp, x_min_exp:x_max_exp]
    dy = 0
    if y_min_exp < 0:
        dy = y_min // 2
    if y_max_exp < y_max:
        dy = (y_max - y_max_exp) // 2
    center = (x_max_exp - x_min_exp) // 2, (y_max_exp - y_min_exp) // 2 + dy

    return cropped, x_min_exp, y_min_exp, x_max_exp, y_max_exp, center


def get_affine_transform(w_a, h_a, fx, fy, angle_degrees):
    """
    计算从 A 到 B 的高精度坐标变换矩阵
    :param w_a, h_a: 图像 A 的尺寸
    :param fx, fy: 缩放因子
    :param angle_degrees: 旋转角度 (OpenCV标准: 逆时针为正)
    :return:
        M (2x3 numpy float64 matrix),
        (w_b, h_b) (Tuple[int, int])
    """
    # 1. 基础参数 (强制 float64)
    cx, cy = w_a / 2.0, h_a / 2.0
    theta = math.radians(angle_degrees)

    # 2. 计算三角函数值
    # Y轴向下坐标系 + 逆时针旋转：
    # 旋转矩阵 R = [[cos, sin], [-sin, cos]]
    cos_val = math.cos(theta)
    sin_val = math.sin(theta)

    # 3. 构建核心变换矩阵 elements (R * S)
    # 我们直接手动展开矩阵乘法结果，避免 numpy 多次 matmul 带来的微小误差
    # M_core = [[ a, b, ...], [ c, d, ...]]
    # a = fx * cos, b = fy * sin
    # c = -fx * sin, d = fy * cos
    a = fx * cos_val
    b = fy * sin_val
    c = -fx * sin_val
    d = fy * cos_val

    # 4. 计算变换后的四个角点以确定包围盒
    # 原始角点相对于中心 (cx, cy) 的坐标
    # 这种方式比 transform 整个坐标数组更快且精度更高
    corners_local = np.array([
        [-cx, -cy],  # 左上
        [w_a - cx, -cy],  # 右上
        [w_a - cx, h_a - cy],  # 右下
        [-cx, h_a - cy]  # 左下
    ], dtype=np.float64)

    # 应用核心线性变换 (R * S * point)
    # x_new = a*x + b*y
    # y_new = c*x + d*y
    x_transformed = corners_local[:, 0] * a + corners_local[:, 1] * b
    y_transformed = corners_local[:, 0] * c + corners_local[:, 1] * d

    # 5. 计算高精度边界 (使用 floor 和 ceil)
    # 这是精度提升的关键！不要用 round。
    # 任何小于 0 的部分（例如 -0.01）都必须归入 -1 坐标
    min_x = np.floor(np.min(x_transformed))
    min_y = np.floor(np.min(y_transformed))
    max_x = np.ceil(np.max(x_transformed))
    max_y = np.ceil(np.max(y_transformed))

    w_b = int(max_x - min_x)
    h_b = int(max_y - min_y)

    # 6. 计算最终的平移量 tx, ty
    # 我们需要把 (min_x, min_y) 移到 (0, 0)
    # 之前我们是相对于“中心”算的坐标，现在要转回绝对坐标
    # 核心公式推导：
    # final_x = (a*x_local + b*y_local) - min_x
    # 其中 x_local = x_input - cx
    # 展开：final_x = a*x_input + b*y_input + (-a*cx - b*cy - min_x)

    tx = -a * cx - b * cy - min_x
    ty = -c * cx - d * cy - min_y

    # 7. 组装最终矩阵
    M = np.array([
        [a, b, tx],
        [c, d, ty]
    ], dtype=np.float64)

    return M, (w_b, h_b)


def save_register_stats(arr, save_path=f'results/registration_stats.csv'):
    Path(save_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    columns = ['fid', 'n_inliers', 'ratio', 'prj_error']
    df = pd.DataFrame(arr, columns=columns)
    df.to_csv(save_path, index=False, sep=' ', float_format='%.3f')
    print(f"Saved registration statistics to {save_path}")


if __name__ == '__main__':
    save_register_stats(np.eye(3))
    show_images('output')
