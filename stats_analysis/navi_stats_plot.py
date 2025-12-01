import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

# print(mpl.get_cachedir())

# 1. 设置中文字体 (避免乱码)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plot_bar_line_with_stats(arr, x=None, title=None, figsize=(10, 6),
                             bar_color='#4C72B0', line_color='#DD8452',
                             mean_color='k', rmse_color='r'):
    """
    绘制：柱状图 + 折线图，并显示 mean/median/std/sem/mae。

    参数
    ----
    arr : 1D numpy array
    x : 可选的 x 坐标（与 arr 长度相同）。若为 None 则使用 range(len(arr))。
    title : 图标题
    figsize : 图大小
    其余颜色参数可自定义

    返回
    ----
    fig, ax
    """
    arr = np.asarray(arr).ravel()
    if arr.size == 0:
        raise ValueError("arr 不能为空")

    n = arr.size
    if x is None:
        x = np.arange(n)
    else:
        x = np.asarray(x)
        if x.shape[0] != n:
            raise ValueError("x 和 arr 长度必须一致")

    # 统计量
    mean = float(np.mean(arr))
    rmse = np.sqrt(np.mean(arr ** 2))
    std = np.std(arr)
    sem = float(std / np.sqrt(n)) if n > 0 else 0.0  # 均值的标准误

    # 绘图
    fig, ax = plt.subplots(figsize=figsize)
    # 柱状图
    bars = ax.bar(x, arr, color=bar_color, alpha=0.9, label='value (bar)')

    # 折线图（连线 + 圆点）
    ax.plot(x, arr, marker='o', linestyle='-', color=line_color, label='value (line)')

    # 平均线与中位线
    ax.axhline(mean, color=mean_color, linestyle='-.', linewidth=1.2, label=f"Mean = {mean:.2f}")
    ax.axhline(rmse, color=rmse_color, linestyle='--', linewidth=1.5, label=f"RMSE = {rmse:.2f}")

    # 标准差带 mean ± std（浅色）
    ax.fill_between(x, mean - std, mean + std, color=mean_color, alpha=0.12, label=f"±1 Std ({std:.2f})")

    # 标准误带 mean ± sem（更窄、更透明）
    ax.fill_between(x, mean - sem, mean + sem, color=mean_color, alpha=0.18, hatch='////', label=f"±SEM ({sem:.2f})")

    # 可选：在每个柱子上显示数值
    for rect, val in zip(bars, arr):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., h,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, rotation=0)

    # 统计信息文本框（右上角）
    stats_text = (
        f"N = {n}\n"
        f"Mean = {mean:.2f}\n"
        f"RMSE = {rmse:.2f}\n"
        f"Std = {std:.2f}\n"
        f"SEM = {sem:.2f}\n"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    if title:
        ax.set_title(title)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()
    return fig, ax


if __name__ == '__main__':
    est_traj_file = 'traj_esekf.csv'
    gt_traj_file = 'traj_gt.csv'
    arr1 = np.loadtxt(est_traj_file)
    arr2 = np.loadtxt(gt_traj_file)
    diff = arr1 - arr2
    diff = diff[:, 1:4]
    dist = np.linalg.norm(diff, axis=1)

    fig, ax = plot_bar_line_with_stats(dist, title='ESEKF视觉导航误差统计')
    save_path = 'navi_stats_plot.png'
    fig.savefig(save_path, dpi=300)
    print(f'saved to {save_path}')
