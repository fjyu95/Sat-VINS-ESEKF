import os
import sys
from pathlib import Path

import cv2
import numpy as np

np.set_printoptions(suppress=True)

import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol
import pymap3d as pm
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtGui import QImage

from utils.geo_processing import search_closest_pyramid_level, get_pyramid_resolutions, get_ground_resolutions, \
    get_xy_resolutions
from visual_localization.solve_uav import solve_uav

from ESEKF_IMU.ekf_fusion import init_estimator, ekf_predict, ekf_update, load_imu_parameters
from utils.core_utils import Logger, timer, save_register_stats
from utils.parser import parse_txt_gbk, parse_imu_data, parse_gps_data, parse_timestamp, get_all_gps_gt_enu
from visual_localization.geo_registration import generate_matched_points, get_rotation_angle

wgs842Mercator_trans = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


class EKFWorker(QObject):
    plane_position_signal = pyqtSignal(float, float, int)
    draw_matches_signal = pyqtSignal(QImage, float, float, float, float, int)
    pnp_line_signal = pyqtSignal(float, float, float, float, int)
    ekf_progress_signal = pyqtSignal(int, int)
    ekf_worker_finished_signal = pyqtSignal(bool)

    def __init__(self, config):
        super().__init__()
        self._is_running = True  # 控制整个线程是否结束
        self._is_paused = False  # 控制是否暂停

        self.config = config
        self.images_dir = self.config['aerial_images_dir']
        self.pyramid_dir = self.config['pyramid_images_dir']
        self.dem_path = self.config['dem_path']
        self.basemap_path = self.config['basemap_path']
        self.tif_name_format = self.config['tif_name_format']

        # self.x_res, self.y_res = self.config['x_resolution'], self.config['y_resolution']
        # self.target_res = (self.x_res + self.y_res) / 2
        self.theta = self.config['rotation_angle']

        self.kwargs_register = {'crop_scale': self.config['scale_factor_crop_pyramid'],
                                'reproj_thresh': self.config['register_ransacReprojThreshold'],
                                'min_inliers': self.config['register_min_inliers'],
                                'min_ratio': self.config['register_min_ratio'],
                                'north_bias': self.config['camera_axis_north_bias'],
                                'east_bias': self.config['camera_axis_east_bias'],
                                }

        self.kwargs_pnp = {'camera_intrinsics': self.config['camera_intrinsics'],
                           'camera_distortion': self.config['camera_distortion'],
                           'pnp_ransac_reprojectionError': self.config['pnp_ransac_reprojectionError'],
                           'pnp_ransac_confidence': self.config['pnp_ransac_confidence'],
                           'pnp_ransac_iterations': self.config['pnp_ransac_iterations'],
                           'PnPRefineLM_min_inliers': self.config['PnPRefineLM_min_inliers'],
                           'PnPRefineLM_maxIterations': self.config['PnPRefineLM_maxIterations']}

        self.kwargs_ekf = {'ekf_sigma_p': self.config['ekf_sigma_p'],
                           'ekf_sigma_q': self.config['ekf_sigma_q'],
                           'distance_threshold': self.config['ekf_fusion_threshold']}

        files = sorted(Path(self.images_dir).glob('*.tif'))
        start_idx = self.config['start_frame_idx']
        end_idx = self.config['end_frame_idx']
        if end_idx == -1:
            end_idx = len(files) - 1

        imu_parameters = load_imu_parameters('configs/imu_parameters.yaml')
        record0 = parse_txt_gbk(str(files[start_idx].with_suffix('.txt')))
        self.geo0 = lat0, lon0, height0 = record0['飞机纬度(度)'], record0['飞机经度(度)'], record0['飞机海拔高度(米)']
        p0_ecef = pm.geodetic2ecef(*self.geo0)
        self.t0 = parse_timestamp(record0)

        res_vertical, res_parallel, target_res = get_ground_resolutions(record0, config)
        self.pyramid_resolutions, self.pyramid_sizes = get_pyramid_resolutions(self.pyramid_dir)
        self.best_level, self.best_res = search_closest_pyramid_level(self.pyramid_resolutions, target_res)
        self.top_layer = next(iter(self.pyramid_resolutions))
        self.tile_size = self.pyramid_sizes[self.best_level]
        print(f'selected pyramid level:{self.best_level} resloution:{self.best_res:.2f}m\n')

        if self.theta == -1:
            self.theta = get_rotation_angle(files[start_idx], record0, self.geo0, self.pyramid_dir, self.best_level,
                                            self.best_res, self.tile_size, target_res, self.config)

        tile_north, tile_east = record0['视场中心俯仰角(度)'], record0['视场中心横滚角(度)']
        self.x_res, self.y_res = get_xy_resolutions(res_vertical, res_parallel, tile_north, tile_east, self.theta)

        with rasterio.open(self.basemap_path) as src:
            self.basemap_transform = src.transform

        self.basemap = cv2.imread(self.basemap_path, -1)

        self.p0_enu = parse_gps_data(p0_ecef, self.geo0, self.t0, record0)
        imu_data = parse_imu_data(record0, t0=self.t0)
        self.ekf_estimator, self.sigma_measurement = init_estimator(self.p0_enu, imu_data, imu_parameters,
                                                                    self.kwargs_ekf['ekf_sigma_p'],
                                                                    self.kwargs_ekf['ekf_sigma_q'])
        self.traj_est = [self.p0_enu[:8]]

        self.files = files[start_idx: end_idx + 1]
        if self.config['final_files']:
            self.final_files = self.config['final_files'][start_idx: end_idx + 1]
        else:
            self.final_files = self.files

        if self.config['DEBUG']:
            # files = files[::2]
            num = 30
            self.files = self.files[:num]
            self.final_files = self.final_files[:num]

        self.n_files = len(self.files)

        self.n_vio = 0
        self.i = 1  # 当前待处理帧索引

    def pause(self, paused: bool):
        """设置暂停状态"""
        self._is_paused = paused
        if paused:
            print(">>> 请求暂停")
        else:
            print(">>> 请求继续")

    def stop(self):
        """彻底停止线程（用于关闭窗口时）"""
        self._is_running = False
        print(">>> 请求停止 EKF 线程")

    def emit_position_signal(self, lon, lat):
        x, y = wgs842Mercator_trans.transform(lon, lat)
        row, col = rowcol(self.basemap_transform, x, y)
        self.plane_position_signal.emit(col, row, self.i)

    def post_process_register(self, match_results, img):
        pts_2d, pts_3d_ecef, pts2d_satellite, homo_img2sat, xs, ys, rot_img, match_stats, vis = match_results

        rows, cols = rowcol(self.basemap_transform, xs, ys)
        pts2d_bg = np.column_stack([cols, rows])
        affine, _ = cv2.estimateAffine2D(pts2d_satellite, pts2d_bg)
        homo_sat2bg = np.vstack([affine, [0, 0, 1]])
        homo_img2bg = np.dot(homo_sat2bg, homo_img2sat)

        img_height, img_width = self.basemap.shape[:2]
        img_in_bg = cv2.warpPerspective(img, homo_img2bg, (img_width, img_height))
        x_min, y_min, w_bg, h_bg = cv2.boundingRect(img_in_bg)
        cx, cy = x_min + w_bg / 2, y_min + h_bg / 2

        translation_matrix = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ])
        homo_patch_bg = translation_matrix @ homo_img2bg

        h, w = img.shape[:2]
        img_with_border = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img_with_border, (0, 0), (w - 1, h - 1), (255, 255, 0), thickness=100)  # 画边框
        warped_patch = cv2.warpPerspective(img_with_border, homo_patch_bg, (w_bg, h_bg))

        if warped_patch.ndim == 2:  # 如果是灰度
            warped_patch = cv2.cvtColor(warped_patch, cv2.COLOR_GRAY2BGRA)
        else:  # 如果是 BGR
            warped_patch = cv2.cvtColor(warped_patch, cv2.COLOR_BGR2BGRA)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:] = 255  # 有效区域掩码，透明度交给setOpacity
        warped_mask = cv2.warpPerspective(mask, homo_patch_bg, (w_bg, h_bg))
        warped_patch[:, :, 3] = warped_mask

        h, w, ch = warped_patch.shape
        qimg = QImage(warped_patch.data, w, h, w * ch, QImage.Format_RGBA8888).copy()
        self.draw_matches_signal.emit(qimg, x_min, y_min, w_bg, h_bg, self.i)

        return cx, cy

    def run_ekf_step(self, path):
        purename = Path(path).stem
        txt_path = path.with_suffix('.txt')
        record = parse_txt_gbk(str(txt_path))

        imu_data = parse_imu_data(record, t0=self.t0)
        flight_time = imu_data[0]

        frame_pose = ekf_predict(self.ekf_estimator, imu_data)
        p_enu_imu = frame_pose[1:4]
        imu_geo = imu_lat, imu_lon, _ = geo_pred = pm.enu2geodetic(*p_enu_imu, *self.geo0)
        print(f'p_enu_imu={p_enu_imu}')

        img = cv2.imread(self.final_files[self.i].as_posix(), -1)
        match_results = generate_matched_points(img, record, geo_pred, self.x_res, self.y_res, self.theta,
                                                self.pyramid_dir, self.best_level, self.best_res, self.tile_size,
                                                self.tif_name_format, self.dem_path, purename, **self.kwargs_register)

        if match_results is None:
            self.emit_position_signal(imu_lon, imu_lat)
            self.ekf_progress_signal.emit(self.i, self.n_files - 1)
            print(f'Frame {self.i}/{self.n_files - 1}: matching failed, using IMU only.\n')
            return frame_pose, None

        pts_2d, pts_3d_ecef, pts2d_satellite, homo_img2sat, xs, ys, rot_img, match_stats, vis = match_results
        cx, cy = self.post_process_register(match_results, img)

        pnp_results = solve_uav(pts_2d, pts_3d_ecef, **self.kwargs_pnp)
        if pnp_results is None:
            self.emit_position_signal(imu_lon, imu_lat)
            self.ekf_progress_signal.emit(self.i, self.n_files - 1)
            # draw_solver_results(purename, vis, 'solve pnp failed', f'pnp_err', flag=False)
            return frame_pose, match_stats

        p_ecef, pnp_stats = pnp_results
        enu_data = parse_gps_data(p_ecef, self.geo0, self.t0, record)
        p_enu_pnp = enu_data[1:4]

        bias = np.abs(p_enu_pnp - p_enu_imu)
        dist = np.linalg.norm(bias)
        print(f'visual_localization result: bias with IMU={bias} dist={dist:.2f}m')

        dist_threshold = self.kwargs_ekf['distance_threshold']  # 200m*3倍误差*2方向
        if dist > dist_threshold:
            self.emit_position_signal(imu_lon, imu_lat)
            self.ekf_progress_signal.emit(self.i, self.n_files - 1)
            return frame_pose, match_stats  # 视觉定位不可靠时，使用IMU预测结果

        frame_pose = ekf_update(self.ekf_estimator, enu_data, self.sigma_measurement)
        ekf_enu = frame_pose[1:4]

        ekf_geo = ekf_lat, ekf_lon, ekf_height = pm.enu2geodetic(*ekf_enu, *self.geo0)
        x, y = wgs842Mercator_trans.transform(ekf_lon, ekf_lat)
        row, col = rowcol(self.basemap_transform, x, y)
        self.plane_position_signal.emit(col, row, self.i)
        self.pnp_line_signal.emit(col, row, cx, cy, self.i)  # 使用了VIO才画线
        self.ekf_progress_signal.emit(self.i, self.n_files - 1)
        self.n_vio += 1

        print(f'Frame {self.i}/{self.n_files - 1}: ekf_lat={ekf_lat}, ekf_lon={ekf_lon}, ekf_height={ekf_height:.2f}\n')

        return frame_pose, match_stats

    def save_trajectory(self, ekf_save_path='results/traj_esekf.csv', gt_save_path='results/traj_gt.csv'):
        Path(ekf_save_path).resolve().parent.mkdir(parents=True, exist_ok=True)
        traj_est = np.array(self.traj_est)
        np.savetxt(ekf_save_path, traj_est)
        print(f'ekf trajectory saved to {ekf_save_path}')

        traj_gt = get_all_gps_gt_enu(self.files, self.geo0, self.t0)
        np.savetxt(gt_save_path, traj_gt)
        print(f'ground truth trajectory saved to {gt_save_path}')

    @timer
    def pipeline(self):
        lat0, lon0, _ = self.geo0
        x0, y0 = wgs842Mercator_trans.transform(lon0, lat0)
        row, col = rowcol(self.basemap_transform, x0, y0)
        self.plane_position_signal.emit(col, row, 0)
        print(f'Tasks number: {self.n_files - 1}')

        stats_register = []
        while self._is_running:
            if self._is_paused:
                QThread.msleep(100)
                continue

            path = self.files[self.i]
            frame_pose, match_stats = self.run_ekf_step(path)
            self.traj_est.append(frame_pose)

            if match_stats is not None:
                fid = Path(path).stem
                stats_register.append((fid,) + match_stats)

            self.i += 1
            if self.i == self.n_files:
                break

        ekf_ratio = self.n_vio / (self.n_files - 1) * 100
        save_register_stats(np.array(stats_register))
        self.save_trajectory()
        self.ekf_worker_finished_signal.emit(True)
        print(f'EKF estimation finished, vio ratio:{self.n_vio}/{self.n_files - 1}={ekf_ratio:.2f}%\n')


if __name__ == '__main__':
    # 替换标准输出和错误输出
    sys.stdout = Logger(log_dir='logs', also_print=True)
    sys.stderr = sys.stdout
