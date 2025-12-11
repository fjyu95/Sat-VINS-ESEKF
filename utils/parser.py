import codecs
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
import pymap3d as pm

from utils.config import restore_config


def parse_txt_gbk(path: str) -> dict:
    """
    Parse GBK/GB18030 encoded file into dict.
    """
    record = {}
    # 用 GB18030（向下兼容 GBK）打开
    with codecs.open(path, 'r', encoding='gb18030') as f:
        for line in f:
            if ':' not in line:
                continue
            key, value = line.split(':', maxsplit=1)
            key = key.strip()
            value = value.strip()

            # 数值转换
            try:
                value = float(value) if '.' in value else int(value)
            except ValueError:
                pass  # 保留字符串
            record[key] = value
    return record


def load_config(path="configs/base.yaml"):
    if not os.path.exists(path):
        path = "configs/default.yaml"

    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print("Error loading config file: ", e)
        return None

    root_dir = config['dataset_root_dir']
    root_dir = Path(root_dir).resolve().as_posix()
    for k, v in config.items():
        if k.endswith('_dir') or k.endswith('_path'):
            config[k] = os.path.join(root_dir, config[k])

    return config


def parse_timestamp(meta_data=None, path=None):
    if meta_data is None:
        meta_data = parse_txt_gbk(path)

    date_str, time_str = meta_data['日期(年.月.日)'], meta_data['日期(时.分.秒.毫秒)']
    dt_str = f'{date_str} {time_str}'
    dt = datetime.strptime(dt_str, "%Y.%m.%d %H:%M:%S:%f")
    timestamp = dt.timestamp()

    return timestamp


def get_geodetic_gt(record: dict):
    lon = record['飞机经度(度)']
    lat = record['飞机纬度(度)']
    height = record['飞机海拔高度(米)']
    return np.array([lat, lon, height])


def parse_imu_data(meta_data=None, path=None, t0=.0):
    if meta_data is None:
        meta_data = parse_txt_gbk(path)

    ang1 = meta_data['飞机航向角速率(度/秒)']  # yaw
    ang2 = meta_data['飞机俯仰角速度(度/秒)']  # pitch
    ang3 = meta_data['飞机横滚角速度(度/秒)']  # roll
    ang1, ang2, ang3 = [math.radians(x) for x in [ang1, ang2, ang3]]  # 转弧度

    acc1 = meta_data['东向加速度(米/秒平方)']  # x
    acc2 = meta_data['北向加速度(米/秒平方)']  # y
    acc3 = meta_data['天向加速度(米/秒平方)']  # z

    ts = parse_timestamp(meta_data) - t0
    imu_data = np.array([ts, ang3, ang2, ang1, acc1, acc2, acc3])

    max_val = np.abs(imu_data[1:]).max()
    if max_val < 1e-6:
        imu_data[1:4] += 1e-7  # 防止除0

    return np.array(imu_data)


def parse_gps_data(p_ecef, p0, t0, meta_data=None, path=None):
    if meta_data is None:
        meta_data = parse_txt_gbk(path)  # 解析过的直接用

    p_enu = pm.ecef2enu(*p_ecef, *p0)
    roll, pitch, yaw = meta_data['飞机横滚角(度)'], meta_data['飞机俯仰角(度)'], meta_data['飞机航向角(度)']
    v_E, v_N, v_U = meta_data['飞机东速(米/秒)'], meta_data['飞机北速(米/秒)'], meta_data['飞机天速(米/秒)']
    angles = np.array([roll, pitch, yaw])

    # 定义欧拉角（单位：弧度），顺序为Z-Y-X（yaw, pitch, roll）
    r = R.from_euler('xyz', angles, degrees=True)
    q = r.as_quat(scalar_first=True)

    v = np.array([v_E, v_N, v_U])
    ts = parse_timestamp(meta_data) - t0
    gps_data = np.concatenate(([ts], p_enu, q, v, angles))  # 1+3+4+3+3+3=17

    return gps_data


def get_all_gps_gt_enu(files, p0, t0):
    gt_enu = []
    for path in files:
        path = path.with_suffix('.txt').as_posix()
        record = parse_txt_gbk(path)
        geo_pos = record['飞机纬度(度)'], record['飞机经度(度)'], record['飞机海拔高度(米)']
        p_ecef = pm.geodetic2ecef(*geo_pos)
        data = parse_gps_data(p_ecef, p0, t0, record)
        gt_enu.append(data[:8])

    gt_enu = np.array(gt_enu)

    return gt_enu


if __name__ == '__main__':
    pass
