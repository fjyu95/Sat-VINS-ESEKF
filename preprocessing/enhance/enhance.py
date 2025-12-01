import os
from pathlib import Path

import cv2
import numpy as np
from skimage import exposure, io
from skimage.util import img_as_ubyte


def gamma_enhance(img, gamma=0.5):
    lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(img, lookup_table)

    return enhanced


def clahe_enhance(
        path: str,
        method="opencv",
        clip_limit=2.0,
        tile_grid=(8, 8),
        save_dir='enhanced',
        save_enhanced=False,
        save_combined=False,
):
    img = cv2.imread(path, -1)

    # 自动判断图像是否为灰度/RGB
    if img.ndim == 2:
        gray = True
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = False
    else:
        raise ValueError("输入图像必须为灰度或 RGB")

    # 转换为适合 skimage 或 OpenCV 的格式
    if method == "opencv":
        # OpenCV CLAHE：只支持 8bit
        if img.dtype != np.uint8:
            img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            img8 = img.copy()

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

        if gray:
            enhanced = clahe.apply(img8)
        else:
            channels = cv2.split(img8)
            enhanced = cv2.merge([clahe.apply(c) for c in channels])

            # # 转换到 LAB 空间
            # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            # l, a, b = cv2.split(lab)
            # l2 = clahe.apply(l)
            #
            # lab2 = cv2.merge([l2, a, b])
            # enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    elif method == "skimage":
        # img = io.imread(path) # RGB channel order
        if img.dtype != np.float32 and img.dtype != np.float64:
            imgf = img.astype(np.float32) / 255.0
        else:
            imgf = img.copy()

        enhanced_f = exposure.equalize_adapthist(imgf, clip_limit=clip_limit / 100.0, kernel_size=tile_grid)  # 速度略慢
        # enhanced = (enhanced * 255).astype(np.uint8)
        enhanced = img_as_ubyte(enhanced_f)
    else:
        raise ValueError("method 必须是 'opencv' 或 'skimage'")

    if save_enhanced or save_combined:
        os.makedirs(save_dir, exist_ok=True)

    if save_enhanced:
        save_path = os.path.join(save_dir, f'enhanced_{path.stem}.jpg')
        cv2.imwrite(save_path, enhanced)

    if save_combined:
        save_path_concat = os.path.join(save_dir, f'combined_{path.stem}.jpg')
        combined = np.hstack((img, enhanced))
        cv2.imwrite(save_path_concat, combined)

    return img, enhanced


def test_enhance():
    data_dir = 'test_data'
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
    img_list = [p for p in Path(data_dir).rglob("*") if p.suffix.lower() in exts]

    for i, path in enumerate(img_list, start=1):
        clahe_enhance(path, save_enhanced=True, save_combined=True)
        print(i, path)


if __name__ == '__main__':
    test_enhance()
