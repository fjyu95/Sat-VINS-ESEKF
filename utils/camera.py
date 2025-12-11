import os
import math
from pathlib import Path

import cv2
import numpy as np
import pymap3d as pm

from utils.core_utils import extrinsic_to_c2w, timer
from utils.parser import load_config, parse_txt_gbk
from visual_localization.solve_uav import lm_refine_pnp

os.chdir('../')
config_path = 'configs/base.yaml'
config = load_config(config_path)


@timer
def iter_search_focal(pts_2d, pts_3d, ecef_xyz, K, dist_coeffs, factor=0.2, n_iters=100):
    n_pts = len(pts_2d)
    print(f'{n_pts} points to solve PnP')

    fx, _, cx, _, fy, cy = K[:2].flatten()
    f_init = (fx + fy) / 2
    image_width, image_height = cx * 2, cy * 2
    d = max(image_width, image_height)
    fov = 2 * math.atan(d / (2 * f_init)) * 180 / math.pi

    min_fov = fov * (1 - factor)
    max_fov = fov * (1 + factor)

    # fov_range = np.array([min_fov, max_fov])
    # focal_range = d / (2 * np.tan(np.deg2rad(fov_range / 2)))  # w/2f = tan(fov/2)
    # focal_mean = np.mean(focal_range)

    # 定义焦距搜索范围和步长
    fov_steps = np.linspace(min_fov, max_fov, num=n_iters)
    focal_length_step = d / (2 * np.tan(np.deg2rad(fov_steps / 2)))
    n_steps = len(focal_length_step)
    print(f"range of focal lengths: {focal_length_step.min(), focal_length_step.max()},len={n_steps}")

    # --- 迭代搜索 ---
    best_focal_length = f_init
    best_distance = float('inf')
    best_rvec = None
    best_tvec = None

    for i, f_guess in enumerate(focal_length_step, start=1):
        # 1. 构建临时的相机矩阵
        K = np.array([
            [f_guess, 0, cx],
            [0, f_guess, cy],
            [0, 0, 1]
        ])

        # 2. 求解PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, K, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=config['pnp_ransac_reprojectionError'],  # 10
            confidence=config['pnp_ransac_confidence'],  # 0.95
            iterationsCount=config['pnp_ransac_iterations']  # 500
        )

        if not success:
            return None

        n_inliers = inliers.shape[0]
        ratio = n_inliers / n_pts
        if n_inliers > config['PnPRefineLM_min_inliers']:
            rvec_lm, tvec_lm = lm_refine_pnp(pts_3d, pts_2d, inliers, K, rvec, tvec, dist_coeffs,
                                             max_iterations=config['PnPRefineLM_maxIterations'])
            rvec, tvec = rvec_lm, tvec_lm

        # 3. 计算重投影误差
        projected_points, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist_coeffs)
        projected_points = projected_points.reshape(-1, 2)
        # print(f"PnP 成功, inliers: {n_inliers}, ratio: {ratio:.3f}, prj_error: {prj_error:.3f}")

        pred_xyz = extrinsic_to_c2w(rvec, tvec)
        diff = pred_xyz - ecef_xyz
        dist = np.linalg.norm(diff)

        # 4. 更新最佳结果
        if dist < best_distance:
            best_focal_length = f_guess
            best_distance = dist
            best_rvec = rvec
            best_tvec = tvec

            fov = 2 * math.atan(d / (2 * f_guess)) * 180 / math.pi
            prj_error = np.median(np.linalg.norm(pts_2d - projected_points, axis=1))
            print(
                f"i = {i}, focal_length {f_guess:.2f}, distance {dist:.1f}, prj_error {prj_error:.2f}, fov {fov:.3f}°")

    best_fov = 2 * math.atan(d / (2 * best_focal_length)) * 180 / math.pi

    print(f"最佳焦距估计: {best_focal_length} 最佳视场角: {best_fov:.3f}°")
    print(f"最小距离误差: {best_distance}")
    print(f"对应的旋转向量 (rvec): \n{best_rvec}")
    print(f"对应的平移向量 (tvec): \n{best_tvec}")

    return best_focal_length, best_rvec, best_tvec, best_fov


@timer
def batch_search_focal(data_dir='image_matches_results'):
    K = np.eye(3)
    K[:2] = np.array([246875, 0, 2372, 0, 246875, 1897]).reshape(2, -1)
    distort = np.zeros(5)

    files = sorted(Path(data_dir).glob('*.npy'))
    # files = files[::5]
    n_files = len(files)

    all_focal_guess = []
    for i, path in enumerate(files, start=1):
        pure_name = path.stem
        fid = '_'.join(pure_name.split('_')[2:])
        arr = np.load(path)
        pts_2d, pts_3d = arr[:, :2], arr[:, 2:]
        pts_3d = np.ascontiguousarray(pts_3d)

        txt_path = os.path.join(config['aerial_images_dir'], f'{fid}.txt')
        record = parse_txt_gbk(txt_path)
        p_geo = record['飞机纬度(度)'], record['飞机经度(度)'], record['飞机海拔高度(米)']
        p_ecef = pm.geodetic2ecef(*p_geo)

        ret = iter_search_focal(pts_2d, pts_3d, p_ecef, K, distort, n_iters=20)
        if ret is None:
            print(f'{i}/{n_files} failed to guess focal length for {fid}')
            continue

        focal_guess, *_ = ret
        all_focal_guess.append(focal_guess)
        print(f"{i}/{n_files}: {focal_guess:.2f}")

    all_focal_guess = np.array(all_focal_guess)
    print(f'\nfinished all focal guess.')
    print(np.round(all_focal_guess, 2))

    focal_mean = np.mean(all_focal_guess)
    focal_median = np.median(all_focal_guess)
    print(f"best focal mean: {focal_mean:.2f}, best focal median: {focal_median:.2f}")

    return focal_median


def test():
    pass


if __name__ == "__main__":
    batch_search_focal()
