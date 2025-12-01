import math
import os
from pathlib import Path

import cv2
import numpy as np
import rasterio
import pymap3d as pm
import pyproj
from pyproj import Transformer, CRS
from rasterio.transform import rowcol
from rasterio.merge import merge
import rasterio.windows as windows

wgs84_sys = 'epsg:4326'  # wgs84
wgs3d_sys = 'epsg:4979'  # wgs84 3d
ecef_sys = 'epsg:4978'  # ECEF X,Y,Z [m]

# WGS84椭球模型常数
WGS84_A = 6378137.0  # 地球半长轴 (米)
WGS84_F = 1 / 298.257223563  # 扁率
WGS84_E2 = WGS84_F * (2 - WGS84_F)  # 偏心率的平方,2f-f^2


def ecef2wgs84(arr):
    x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
    lla = pm.ecef2geodetic(x, y, z)
    ret = np.array(lla).T
    return ret


def wgs842ecef(arr):
    lon, lat, alt = arr[:, 0], arr[:, 1], arr[:, 2]
    xyz = pm.geodetic2ecef(lat, lon, alt)
    ret = np.array(xyz).T
    return ret


def enu_to_ned(enu):
    # ENU to NED:
    # ENU: [e, n, u] → NED: [n, e, -u]
    return np.array([enu[1], enu[0], -enu[2]])


def get_geotiff_transform(path: str) -> tuple:
    with rasterio.open(path) as src:
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs
    return transform, width, height, crs


def crop_from_whole_basemap(whole_basemap_path, lon, lat, dw, dh):
    with rasterio.open(whole_basemap_path) as src:
        row, col = rowcol(src.transform, lon, lat)
        res_x, res_y = src.res

    patch_height, patch_width = map(lambda t: math.ceil(t / res_y), [dh, dw])
    img = cv2.imread(whole_basemap_path)
    cropped = img[row:row + patch_height, col:col + patch_width]
    return cropped, src.transform


def get_pyramid_resolutions(pyramid_path):
    """
    从金字塔目录（含 L11, L12 ...）中读取每层.tif文件的空间分辨率。
    自动过滤非标准目录，并按数字层级排序。
    """
    resolutions = {}
    sizes = {}
    sub_dirs = sorted(Path(pyramid_path).glob('L[0-9][0-9]'), key=lambda p: int(p.name[1:]))

    for level_path in sub_dirs:
        tif_files = [f for f in os.listdir(level_path) if f.lower().endswith(".tif")]
        if not tif_files:
            continue

        tif_path = os.path.join(level_path, tif_files[0])
        level = int(level_path.name[1:])
        with rasterio.open(tif_path) as src:
            res_x = src.transform.a  # 像素宽度
            res_y = -src.transform.e  # 像素高度（通常为负数）
            mean_res = (res_x + res_y) / 2
            resolutions[level] = mean_res
            sizes[level] = src.shape[0]

    print(f"Pyramid resolutions:")
    for k, v in resolutions.items():
        print(f"L{k}: {v}m")
    print()

    return resolutions, sizes


def search_closest_pyramid_level(res_dict, target_res):
    """
    根据目标分辨率，返回最匹配的金字塔层级。
    res_dict: dict, 例如 {"L11": 0.597164, "L12": 1.194328, "L13": 2.388656}
               也可为 {"L11": (xres, yres), ...}
    target_res: float, 目标空间分辨率（米/像素）
    """
    best_level = None
    best_diff = float("inf")

    for level, res in res_dict.items():
        # 兼容 (xres, yres) 或单值形式
        if isinstance(res, (tuple, list)):
            mean_res = (abs(res[0]) + abs(res[1])) / 2
        else:
            mean_res = abs(res)

        diff = abs(mean_res - target_res)
        if diff < best_diff:
            best_diff = diff
            best_level = level

    return best_level, res_dict[best_level]


def calculate_ground_coordinates(lat_deg: float,
                                 lon_deg: float,
                                 altitude_m: float,
                                 ground_height: float,
                                 angle_north_deg: float,
                                 angle_east_deg: float,
                                 ) -> tuple[float, float]:
    """
    根据无人机的位置、高度和相机的指向，估算图像中心点的地理坐标。

    此计算基于WGS84椭球模型，假设地面为局部平坦切面进行三角投影，
    并精确计算该纬度下的经纬度转换比例。

    参数:
    drone_lon_deg (float): 无人机GPS的经度 (单位: 度)
    drone_lat_deg (float): 无人机GPS的纬度 (单位: 度)
    angle_north_deg (float): 相机光轴从天顶(垂直向下)向北的倾斜角 (单位: 度)
    angle_east_deg (float): 相机光轴从天顶(垂直向下)向东的倾斜角 (单位: 度)
    altitude_m (float): 无人机距地面的高度 (AGL, Above Ground Level) (单位: 米)

    返回:
    tuple[float, float]: (center_lon_deg, center_lat_deg)
                         图像中心点的估算经纬度 (单位: 度)
    """

    # --- 步骤 1: 将所有角度输入转换为弧度 ---
    lat_rad = math.radians(lat_deg)
    angle_north_rad = math.radians(-(angle_north_deg - 1.75))  # 系统误差修正
    angle_east_rad = math.radians(-angle_east_deg)

    # --- 步骤 2: 计算地面偏移量 (米) ---
    # 使用正切函数计算在地面上的北向和东向偏移
    h = altitude_m - ground_height
    d_north = h * math.tan(angle_north_rad)
    d_east = h * math.tan(angle_east_rad)

    # --- 步骤 3: 计算当前纬度的地球曲率半径 ---
    # (更精确的米到度转换)
    sin_lat = math.sin(lat_rad)
    sin_lat_sq = sin_lat ** 2

    # W = sqrt(1 - e^2 * sin^2(phi))
    w = math.sqrt(1 - WGS84_E2 * sin_lat_sq)

    # 子午圈曲率半径 (用于南北方向)
    # R_M = a * (1 - e^2) / W^3
    r_m = WGS84_A * (1 - WGS84_E2) / (w ** 3)

    # 卯酉圈曲率半径 (用于东西方向)
    # R_N = a / W
    r_n = WGS84_A / w

    # --- 步骤 4: 将米偏移量转换为经纬度偏移量 (度) ---

    # 纬度偏移 (南北方向)
    # delta_lat = d_north / R_M
    delta_lat_deg = math.degrees(d_north / r_m)

    # 经度偏移 (东西方向)
    # 东西方向的圆周半径 R_P = R_N * cos(phi)
    cos_lat = math.cos(lat_rad)

    # 避免在极点附近出现除零错误
    if cos_lat < 1e-9:
        delta_lon_deg = 0.0  # 在极点，经度无意义或变化极大
    else:
        r_p = r_n * cos_lat
        # delta_lon = d_east / R_P
        delta_lon_deg = math.degrees(d_east / r_p)

    # --- 步骤 5: 计算最终的经纬度 ---
    center_lon_deg = lon_deg + delta_lon_deg
    center_lat_deg = lat_deg + delta_lat_deg

    return center_lon_deg, center_lat_deg


def crop_basemap_from_pyramid(root_dir, lon, lat, dw, dh, top_layer, tar_layer, resolution, tile_size):
    """
    1、wgs84经纬度转web墨卡托投影坐标
    2、在塔顶层级获取左上角坐标，并计算坐标偏移量
    3、根据最匹配层级的分辨率，计算瓦片行列号，再根据投影坐标计算出准确像素坐标
    4、根据中心点像素坐标，对边长扩展后，计算航片覆盖的瓦片集合，以及各瓦片的裁剪范围
    5、对各瓦片裁剪后的有效范围进行拼接，返回完整图像
    """

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)  # 投影转换器
    x, y = transformer.transform(lon, lat)  # 中心点的web墨卡托投影坐标
    print(f'estimated center point: {lon:.3f}, {lat:.3f}')

    x, y = x - dw / 2, y + dh / 2  # 左上角投影坐标

    top_tif_path = sorted(Path(root_dir).glob(f'L{top_layer}/*.tif'))[0]
    prefix = '_'.join(top_tif_path.stem.split('_')[:3])
    with rasterio.open(top_tif_path) as src:
        x0, y0 = src.xy(0, 0, offset='ul')  # 金字塔左上角坐标
        # x0, y0 = src.transform.c, src.transform.f
        # row, col = rowcol(src.transform, x0, y0)  # 坐标反算

    dx, dy = x - x0, y0 - y
    patch_width, patch_height = map(lambda t: math.ceil(t / resolution), [dw, dh])

    d_dist = tile_size * resolution
    x_tile = int(dx / d_dist) + 1  # 图像索引是从1开始
    y_tile = int(dy / d_dist) + 1

    delta_x_dist = divmod(dx, d_dist)
    delta_y_dist = divmod(dy, d_dist)
    delta_x = int(delta_x_dist[1] / resolution)
    delta_y = int(delta_y_dist[1] / resolution)
    upper_left_dx = tile_size - delta_x
    upper_left_dy = tile_size - delta_y
    rest_x = patch_width - upper_left_dx
    rest_y = patch_height - upper_left_dy
    x_tile_num = math.ceil(rest_x / tile_size) + 1  # 算上左上角的块
    y_tile_num = math.ceil(rest_y / tile_size) + 1

    src_files = []
    for i, row in enumerate(range(y_tile, y_tile + y_tile_num)):
        for j, col in enumerate(range(x_tile, x_tile + x_tile_num)):
            tif_name = f'{prefix}_{row}-{col}.tif'
            path = Path(root_dir) / f'L{tar_layer}' / tif_name

            try:
                reader = rasterio.open(path)
                src_files.append(reader)
            except Exception as e:
                print(f'Error reading {path}: {e}')

    mosaic, mosaic_transform = merge(src_files)
    mosaic_np = cv2.cvtColor(mosaic.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

    win = ((delta_y, delta_y + patch_height), (delta_x, delta_x + patch_width))
    cropped = mosaic_np[delta_y:delta_y + patch_height, delta_x:delta_x + patch_width]
    cropped_transform = windows.transform(win, mosaic_transform)

    return cropped, cropped_transform


def query_DEM(path: str, arr) -> float:
    with rasterio.open(path) as src:
        elev = src.read(1)  # 二维 numpy 数组
        nodata = src.nodatavals[0]  # ALOS 默认 -9999
        scale = src.scales[0] if src.scales[0] else 1.0  # 几乎总是 1.0
        elev_m = elev * scale  # 单位已经是米，乘不乘都一样
        elev_m[elev_m == nodata] = np.nan

    lons, lats = arr[:, 0], arr[:, 1]
    rows, cols = rowcol(src.transform, lons, lats)
    height = elev_m[rows, cols]

    crs_src = "urn:ogc:def:crs,crs:EPSG::4326,crs:EPSG::5773"
    crs_dst = "urn:ogc:def:crs:EPSG::4979"
    transformer = pyproj.Transformer.from_crs(crs_src, crs_dst, always_xy=True)
    out_lon, out_lat, alts = transformer.transform(lons, lats, height)

    return alts


if __name__ == '__main__':
    pass
