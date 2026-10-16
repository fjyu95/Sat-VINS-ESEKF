"""各功能页面共用的 GUI 组件与辅助函数。"""
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QGraphicsView, QGroupBox, QGridLayout, QHBoxLayout, QVBoxLayout,
    QRadioButton, QLineEdit, QPushButton, QFileDialog, QMessageBox, QButtonGroup,
)
from PyQt5.QtGui import QPainter, QFont
from PyQt5.QtCore import Qt

from stats_analysis.image_viewer import ImageBrowser
from stats_analysis.navi_stats_plot import plot_bar_line_with_stats
from stats_analysis.pandas_viewer import PandasViewer
from utils.viz import draw_traj


# ----------------------------------------------------------------------
# 通用控件
# ----------------------------------------------------------------------
class ZoomableGraphicsView(QGraphicsView):
    """滚轮缩放 + 拖拽平移的图像视图。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

    def wheelEvent(self, event):
        scale_factor = 1.25
        if event.angleDelta().y() > 0:
            self.scale(scale_factor, scale_factor)
        else:
            self.scale(1 / scale_factor, 1 / scale_factor)


class StorageSettingGroup(QGroupBox):
    """「存储设定」分组：自动默认路径 / 手动指定路径。供各仿真单元页面复用。"""

    def __init__(self, parent=None):
        super().__init__("存储设定", parent)

        self.radio_auto = QRadioButton("自动储存至默认路径")
        self.radio_auto.setChecked(True)
        self.radio_manual = QRadioButton("手动指定储存路径")

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("D:/some_dir")
        self.path_edit.setEnabled(False)
        self.btn_open = QPushButton("打开...")
        self.btn_open.setEnabled(False)

        self.radio_manual.toggled.connect(self._on_manual_toggled)
        self.btn_open.clicked.connect(self._choose_dir)

        layout = QGridLayout(self)
        layout.addWidget(self.radio_auto, 0, 0, 1, 3)
        layout.addWidget(self.radio_manual, 1, 0)
        layout.addWidget(self.path_edit, 1, 1)
        layout.addWidget(self.btn_open, 1, 2)

    def _on_manual_toggled(self, checked):
        self.path_edit.setEnabled(checked)
        self.btn_open.setEnabled(checked)

    def _choose_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "选择存储路径", self.path_edit.text() or ".")
        if directory:
            self.path_edit.setText(directory)

    def storage_dir(self):
        """返回手动指定的存储路径；自动模式或未填写时返回 None。"""
        if self.radio_manual.isChecked() and self.path_edit.text().strip():
            return self.path_edit.text().strip()
        return None


def make_mode_group(choices=("注入式模式", "仿真式模式"), title="模式选择", parent=None):
    """生成「模式选择」分组，返回 (QGroupBox, [QRadioButton, ...])。"""
    group = QGroupBox(title, parent)
    layout = QHBoxLayout(group)
    button_group = QButtonGroup(group)
    radios = []
    for i, text in enumerate(choices):
        radio = QRadioButton(text)
        if i == 0:
            radio.setChecked(True)
        button_group.addButton(radio)
        radios.append(radio)
        layout.addWidget(radio)
    layout.addStretch(1)
    return group, radios


# ----------------------------------------------------------------------
# 全局界面风格 / 字体
# ----------------------------------------------------------------------
def set_style_fusion():
    QApplication.setStyle("Fusion")


def set_style_windows():
    QApplication.setStyle("Windows")


def set_style_gtk():
    QApplication.setStyle("GTK+")


def set_font_size_small():
    QApplication.setFont(QFont("Microsoft YaHei", 12))


def set_font_size_medium():
    QApplication.setFont(QFont("Microsoft YaHei", 14))


def set_font_size_large():
    QApplication.setFont(QFont("Microsoft YaHei", 16))


# ----------------------------------------------------------------------
# 后置处理 / 统计分析辅助（包裹已有功能，供多个页面复用）
# ----------------------------------------------------------------------
def show_match_results(parent):
    """打开图像配准结果浏览器。"""
    try:
        viewer = ImageBrowser(parent)
        viewer.open_folder('image_matches_results')
        viewer.show()
        return viewer
    except Exception as e:
        print(e)
        QMessageBox.critical(parent, "错误", f"查看图像匹配结果失败！\n{e}", QMessageBox.Ok)
        return None


def show_registration_stats(parent):
    """打开配准统计数据表。"""
    try:
        viewer = PandasViewer(parent, path='results/registration_stats.csv')
        viewer.show()
        return viewer
    except Exception as e:
        print(e)
        QMessageBox.critical(parent, "错误", f"查看图像配准统计数据失败！\n{e}", QMessageBox.Ok)
        return None


def show_est_vs_gt_traj(parent):
    """绘制估计轨迹与真值轨迹对比曲线。"""
    try:
        draw_traj('results/traj_esekf.csv', 'results/traj_gt.csv')
    except Exception as e:
        print(e)
        QMessageBox.critical(parent, "错误", f"显示导航轨迹失败！\n{e}", QMessageBox.Ok)


def compute_ekf_error_stats(parent):
    """统计 ESEKF 导航定位误差并保存误差分布图，返回图片路径或 None。"""
    try:
        est_traj_file = 'results/traj_esekf.csv'
        gt_traj_file = 'results/traj_gt.csv'
        arr1 = np.loadtxt(est_traj_file)
        arr2 = np.loadtxt(gt_traj_file)
        diff = arr1 - arr2
        diff = diff[1:, 1:4]  # 跳过第一行的真值数据
        dist = np.linalg.norm(diff, axis=1)

        fig, ax = plot_bar_line_with_stats(dist, title='ESEKF视觉导航误差统计')
        save_path = 'navi_accuracy.png'
        fig.savefig(save_path, dpi=300)
        print(f'saved to {save_path}')
        return save_path
    except Exception as e:
        print(e)
        QMessageBox.critical(parent, "错误", f"显示视觉导航误差统计失败！\n{e}", QMessageBox.Ok)
        return None
