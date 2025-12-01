import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.core_utils import save_images


def draw_loftr(img1, img2, pts1, pts2):
    H1, W1 = img1.shape[:2]
    H2, W2 = img2.shape[:2]
    canvas = np.zeros((max(H1, H2), W1 + W2, 3), np.uint8)
    canvas[:H1, :W1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    canvas[:H2, W1:] = img2
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        cv2.circle(canvas, (int(x1), int(y1)), 2, (0, 255, 0), -1)
        cv2.circle(canvas, (int(x2) + W1, int(y2)), 2, (0, 255, 0), -1)
        cv2.line(canvas, (int(x1), int(y1)), (int(x2) + W1, int(y2)), (0, 0, 255), 1)
    return canvas


def draw_loftr_matches1(img0_bgr, img1_bgr, kps0, kps1):
    # 2. 将 (N, 2) 的Numpy点转换为OpenCV的KeyPoint对象列表
    cv_kps0 = [cv2.KeyPoint(p[0], p[1], 1) for p in kps0]
    cv_kps1 = [cv2.KeyPoint(p[0], p[1], 1) for p in kps1]

    # 3. 创建DMatch对象列表
    # (queryIdx, trainIdx, distance) - 这里我们只关心索引
    matches = [cv2.DMatch(i, i, 1.0) for i in range(len(kps0))]

    # 4. 使用cv2.drawMatches绘制匹配
    # flags=2: 只绘制匹配点，不绘制单独的关键点，使画面更整洁
    match_img = cv2.drawMatches(
        img0_bgr, cv_kps0,
        img1_bgr, cv_kps1,
        matches, None,
        matchColor=(0, 0, 255),
        singlePointColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return match_img


def draw_matches_fixed_width_centered(img0, img1, pts0=None, pts1=None, conf=None, target_width=640, gap=20):
    # 确保为 RGB 图像
    def ensure_rgb(img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    img0 = ensure_rgb(img0)
    img1 = ensure_rgb(img1)

    # 统一图像宽度为 target_width，等比缩放
    def resize_to_width(img, width):
        h, w = img.shape[:2]
        scale = width / w
        new_size = (width, int(h * scale))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA), scale

    img0_resized, scale0 = resize_to_width(img0, target_width)
    img1_resized, scale1 = resize_to_width(img1, target_width)

    h0, w0 = img0_resized.shape[:2]
    h1, w1 = img1_resized.shape[:2]
    canvas_height = max(h0, h1)
    canvas_width = w0 + w1 + gap

    # 创建画布
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # 图像在高度方向居中对齐
    y_offset0 = (canvas_height - h0) // 2
    y_offset1 = (canvas_height - h1) // 2

    canvas[y_offset0:y_offset0 + h0, :w0] = img0_resized
    canvas[y_offset1:y_offset1 + h1, w0 + gap:] = img1_resized

    if conf is None:
        return canvas

    n = pts0.shape[0]
    # cv2.putText(canvas, f"Matches: {n}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 缩放关键点坐标
    pts0 = np.array(pts0) * scale0
    pts1 = np.array(pts1) * scale1

    # 对 pts1 应用偏移（x 方向 +w0+gap, y 方向 +y_offset1）
    pts0[:, 1] += y_offset0
    pts1[:, 0] += w0 + gap
    pts1[:, 1] += y_offset1

    # 归一化置信度
    conf = np.clip(conf, 0, None)
    conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)

    # 绘制匹配点和连线
    for (x0, y0), (x1, y1), c in zip(pts0, pts1, conf_norm):
        pt0 = (int(x0), int(y0))
        pt1 = (int(x1), int(y1))
        color = tuple((np.array(plt.cm.jet(c)[:3]) * 255).astype(np.uint8).tolist())
        cv2.circle(canvas, pt0, 4, color, -1)
        cv2.circle(canvas, pt1, 4, color, -1)
        cv2.line(canvas, pt0, pt1, color, 1)

    return canvas


def draw_matches_with_conf(img0, img1, pts0, pts1, conf):
    # 创建并拼接图像（横向拼接）
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    height = max(h0, h1)
    canvas = np.zeros((height, w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:] = img1

    # 归一化置信度 [0,1]
    conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)

    # colors = plt.cm.RdYlGn(conf)[:, :3] * 255  # (N,3) 0-255，颜色映射（红→绿）
    # colors = colors.astype(np.uint8)

    colors = plt.cm.jet(conf_norm)[:, :3] * 255  # jet colormap，颜色渐变（红→黄→绿）

    for (x0, y0), (x1, y1), color in zip(pts0, pts1, colors):
        pt1 = (int(x0), int(y0))
        pt2 = (int(x1 + w0), int(y1))  # 注意 img1 是右侧图，x 要加偏移量

        # 画圆点
        cv2.circle(canvas, pt1, 4, color, -1)
        cv2.circle(canvas, pt2, 4, color, -1)

        # 画连线
        cv2.line(canvas, pt1, pt2, color, 1)

    return canvas


def draw_matches_with_conf_scaled(img0, img1, pts0, pts1, conf, gap=20, fixed_width=960):
    # 确保为 RGB 图像
    def ensure_rgb(img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    img0 = ensure_rgb(img0)
    img1 = ensure_rgb(img1)

    # 统一宽度为 img0 的宽度，等比缩放 img1
    w0 = img0.shape[1]
    h0 = img0.shape[0]
    h1 = img1.shape[0]
    w1 = img1.shape[1]
    scale = w0 / w1

    img1_resized = cv2.resize(img1, (w0, int(h1 * scale)), interpolation=cv2.INTER_AREA)

    # 记录缩放前后的高度
    h1_new = img1_resized.shape[0]
    height = max(h0, h1_new)

    # 创建画布并将两图填入（中间留 gap 像素）
    canvas = np.zeros((height, w0 * 2 + gap, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1_new, w0 + gap:] = img1_resized

    # 调整 pts1 的坐标：先缩放，再偏移（w0 + gap）
    pts0 = np.array(pts0)
    pts1 = np.array(pts1) * scale
    pts1[:, 0] += w0 + gap

    # 归一化置信度 [0, 1]
    conf = np.clip(conf, 0, None)
    conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)

    # 画线与点
    for (x0, y0), (x1, y1), c in zip(pts0, pts1, conf_norm):
        pt1 = (int(x0), int(y0))
        pt2 = (int(x1), int(y1))
        color = tuple(np.array(plt.cm.jet(c))[:3] * 255)
        cv2.circle(canvas, pt1, 4, color, -1)
        cv2.circle(canvas, pt2, 4, color, -1)
        cv2.line(canvas, pt1, pt2, color, 1)

    return canvas


def warp_homo_and_overlay(src, dst, homograpy):
    hB, wB = dst.shape[:2]
    warped_imgA = cv2.warpPerspective(src, homograpy, (wB, hB))
    blended = np.zeros((hB, wB, 3), dtype=np.uint8)
    blended[..., 2] = warped_imgA
    blended[..., 1] = dst[..., 1]

    return warped_imgA, blended


def putText_multiline(img, text, org, fontFace, fontScale, color, thickness=1, lineType=cv2.LINE_AA, gap_ratio=0.3):
    """
    gap_ratio: 行间距 = 字符高度 × ratio
    """
    x, y0 = org
    for i, line in enumerate(text.split('\n')):
        (fw, fh), baseline = cv2.getTextSize(line, fontFace, fontScale, thickness)
        y = y0 + i * (fh + baseline + int(fh * gap_ratio))
        cv2.putText(img, line, (x, y), fontFace, fontScale, color, thickness, lineType)


def draw_solver_results(filename, images, text='', prefix_type='err', flag=True):
    blended, matches, warpped = images
    prefix_matches = ['blended', 'wrapped_pairs', 'loftr_match']
    for img, prefix, in zip(images, prefix_matches):
        path = f'{prefix}_{filename}.jpg'
        if not flag:
            path = f'{prefix_type}_{path}'
        cv2.putText(img, text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), thickness=3)
        save_images(path, img)

    return


def draw_loftr_matches(src_img, dst_img, H, text, pts1_in, pts2_in, mconf, filename, is_valid=False):
    warpped, blended = warp_homo_and_overlay(src_img, dst_img, H)
    putText_multiline(blended, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    path1 = f'blended_{filename}.jpg'

    warpped_vis = draw_matches_fixed_width_centered(warpped, dst_img)
    putText_multiline(warpped_vis, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    path2 = f'wrapped_pairs_{filename}.jpg'

    matches_vis = draw_matches_fixed_width_centered(src_img, dst_img, pts1_in, pts2_in, mconf, target_width=640, gap=20)
    putText_multiline(matches_vis, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    path3 = f'loftr_match_{filename}.jpg'

    if not is_valid:
        path1 = f'err_{path1.split(".")[0]}.jpg'
        path2 = f'err_{path2.split(".")[0]}.jpg'
        path3 = f'err_{path3.split(".")[0]}.jpg'

    save_images(path1, blended, 'image_matches_results')
    save_images(path2, warpped_vis, 'image_matches_results')
    save_images(path3, matches_vis, 'image_matches_results')

    return blended, matches_vis, warpped_vis


def draw_traj(est_filepath='traj_esekf.csv', gt_filepath='traj_gt.csv'):
    if not os.path.exists(est_filepath) or not os.path.exists(gt_filepath):
        print(f"Cannot find {est_filepath} or {gt_filepath}")

    # evo_traj tum ./data/traj_esekf_out.txt --ref ./data/traj_gt_out.txt -p
    cmd = f'evo_traj tum {est_filepath} --ref {gt_filepath} -p'
    # cmd = (
    #     f'conda run -n geovio bash -c "'
    #     f'evo_config set plot_linewidth 2.5; '
    #     f'evo_config set plot_reference_linewidth 2; '
    #     f'evo_config set plot_reference_linestyle --; '
    #     f'evo_config set plot_color blue; '
    #     f'evo_config set plot_reference_color red; '
    #     f'evo_traj tum {est_filepath} --ref {gt_filepath} -p; '
    #     f'evo_config reset"'
    # )
    print(f"Running command: {cmd}")
    os.system(cmd)
