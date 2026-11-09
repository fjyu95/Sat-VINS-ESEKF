"""预处理算法无参考图像质量评价指标。

去雾/增强没有真值参考图，指标分为两类：
- 自身统计量（信息熵、平均梯度、对比度）：前后对比反映信息量与清晰度变化
- 相对原图（PSNR、SSIM）：反映处理引入的变化幅度，不代表绝对质量
"""
import cv2
import numpy as np


def to_gray_u8(img):
    """任意输入（灰度/BGR/float）统一为 uint8 灰度图。"""
    if img is None:
        return None
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def image_entropy(gray):
    """信息熵（bit），越高表示信息量越丰富。"""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / gray.size
    return float(-np.sum(p * np.log2(p)))


def average_gradient(gray):
    """平均梯度（Sobel 幅值均值），越高表示边缘/细节越清晰。"""
    f = gray.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(cv2.magnitude(gx, gy)))


def psnr(before, after):
    """峰值信噪比，以原图为伪参考衡量变化幅度（越大变化越小）。

    注意：无真值（GT）时该指标只反映"改了多少"，不代表"改得好不好"。"""
    mse = np.mean((before.astype(np.float64) - after.astype(np.float64)) ** 2)
    if mse <= 1e-10:
        return float('inf')
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim(before, after):
    """结构相似度，以原图为伪参考衡量结构保持程度（1 为完全一致）。

    注意：无真值（GT）时该指标只反映与原图的相似度，不代表绝对质量。"""
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(before, after, data_range=255))
    except Exception as e:
        print(f'ssim failed: {e}')
        return float('nan')


def compute_metrics(before, after):
    """计算一对 原图/结果图 的指标，返回 dict。"""
    b = to_gray_u8(before)
    a = to_gray_u8(after)
    m = {
        'entropy_before': image_entropy(b),
        'entropy_after': image_entropy(a),
        'avg_grad_before': average_gradient(b),
        'avg_grad_after': average_gradient(a),
        'contrast_before': float(b.std()),
        'contrast_after': float(a.std()),
        'psnr': float('nan'),
        'ssim': float('nan'),
    }
    if a.shape == b.shape:
        m['psnr'] = psnr(b, a)
        m['ssim'] = ssim(b, a)
    return m


def _delta(after, before):
    d = after - before
    return f"{before:.3f} → {after:.3f} ({d:+.3f})"


def format_metrics(m):
    """将 compute_metrics 结果格式化为多行中文文本。"""
    lines = [
        f"信息熵:     {_delta(m['entropy_after'], m['entropy_before'])} bit",
        f"平均梯度:   {_delta(m['avg_grad_after'], m['avg_grad_before'])}",
        f"对比度σ:    {_delta(m['contrast_after'], m['contrast_before'])}",
        f"PSNR:       {m['psnr']:.2f} dB  (对原图)",
        f"SSIM:       {m['ssim']:.4f}    (对原图)",
    ]
    if 'time' in m:
        lines.append(f"耗时:       {m['time']:.2f} s")
    lines.append("注: PSNR/SSIM 以原图为参考，表示变化幅度而非真值质量")
    return '\n'.join(lines)


# metrics.csv 列顺序
METRIC_FIELDS = ['file',
                 'entropy_before', 'entropy_after',
                 'avg_grad_before', 'avg_grad_after',
                 'contrast_before', 'contrast_after',
                 'psnr', 'ssim', 'time_sec']
