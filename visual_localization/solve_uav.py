import cv2
import numpy as np
import pymap3d as pm

from utils.core_utils import extrinsic_to_c2w


def lm_refine_pnp(pts_3d, pts_2d, inliers, K, rvec, tvec, dist_coeffs, max_iterations=100):
    # Levenberg-Marquardt refinement of PnP
    rvec_lm, tvec_lm = cv2.solvePnPRefineLM(
        pts_3d[inliers], pts_2d[inliers], K, dist_coeffs, rvec, tvec,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, 1e-6)
    )

    return rvec_lm, tvec_lm


def solve_pnp(pts_2d, pts_3d, K, dist_coeffs, **kwargs):
    # do not filter outliers, leave it to the later self-supervised step
    n_pts = len(pts_2d)
    print(f'{n_pts} points to solve PnP')

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, K, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=kwargs['pnp_ransac_reprojectionError'],  # 10
        confidence=kwargs['pnp_ransac_confidence'],  # 0.95
        iterationsCount=kwargs['pnp_ransac_iterations']  # 500
    )

    if not success:
        return None

    n_inliers = inliers.shape[0]
    ratio = n_inliers / n_pts
    if n_inliers > kwargs['PnPRefineLM_min_inliers']:
        rvec_lm, tvec_lm = lm_refine_pnp(pts_3d, pts_2d, inliers, K, rvec, tvec, dist_coeffs,
                                         max_iterations=kwargs['PnPRefineLM_maxIterations'])
        rvec, tvec = rvec_lm, tvec_lm

    projected_points, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    prj_error = np.median(np.linalg.norm(pts_2d - projected_points, axis=1))
    stats = n_inliers, ratio, prj_error

    print(f"PnP 成功, inliers: {n_inliers}, ratio: {ratio:.3f}, prj_error: {prj_error:.3f}")

    return rvec, tvec, stats


def solve_uav(pts_2d, pts_3d, **kwargs):
    K = np.eye(3)
    K[:2] = np.array(kwargs['camera_intrinsics']).reshape(2, -1)
    distort = np.array(kwargs['camera_distortion'])

    ret = solve_pnp(pts_2d, pts_3d, K, distort, **kwargs)
    if ret is None:
        return None

    rvec, tvec, stats = ret
    stats = np.round(stats, 3)

    pred_ecef = extrinsic_to_c2w(rvec, tvec)
    pred_geo = pm.ecef2geodetic(*pred_ecef)
    pred_geo = np.array(pred_geo)
    print(f'predicted geodetic coordinates: {pred_geo}')

    return pred_ecef, stats
