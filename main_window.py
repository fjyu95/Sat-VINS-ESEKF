"""无人相对导航全数字仿真系统 —— 主仿真界面入口（对应 ui/samples/1_GUI主仿真界面.png）。

界面按「轨迹生成域 / 功能域 / 数据处理域」组织，各按钮打开独立功能子窗口：
- 轨迹生成域：轨迹生成/回放（trajectory_page）
- 功能域：机间链 / 图像·视觉导航 / 惯导·卫星 仿真单元
- 数据处理域：数据预处理层 / 信息融合层 / 后置处理层（data_process_page）

数据预处理与视觉/惯性融合导航核心功能由 NaviWindow 与 DataProcessPage 承载，
处理流程与原有版本一致。
"""
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGroupBox, QVBoxLayout, \
    QHBoxLayout, QPushButton, QMessageBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from utils.config import restore_config
from utils.core_utils import Logger
from ui.pages.context import AppContext
from ui.pages.common import (
    set_style_fusion, set_style_windows, set_style_gtk,
    set_font_size_small, set_font_size_medium, set_font_size_large,
)
from ui.pages.trajectory_page import TrajectoryPage
from ui.pages.datalink_page import DatalinkPage
from ui.pages.image_gen_page import ImageGenPage
from ui.pages.ins_gnss_page import InsGnssPage
from ui.pages.data_process_page import DataPreprocessPage, InfoFusionPage, PostProcessPage
from ui.pages.navi_window import NaviWindow

BUTTON_WIDTH = 260
BUTTON_HEIGHT = 34


class MainWindow(QMainWindow):
    """主仿真界面：三个功能域按钮面板。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("无人相对导航全数字仿真验证环境")
        self.resize(720, 480)

        self.ctx = AppContext()          # 跨页面共享状态
        self._windows = {}               # 子窗口引用
        self.dark_theme = False

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        root.addWidget(self._domain_group("轨迹生成域", [
            ("轨迹生成功能", self.open_trajectory_page),
        ]))
        root.addWidget(self._domain_group("功能域", [
            ("机间链仿真模块", self.open_datalink_page),
            ("图像/视觉导航仿真模块", self.open_image_gen_page),
            ("惯导/卫星导航仿真模块", self.open_ins_gnss_page),
        ]))
        root.addWidget(self._domain_group("数据处理域", [
            ("数据预处理层", self.open_data_preprocess_page),
            ("信息融合层", self.open_info_fusion_page),
            ("后置处理层", self.open_post_process_page),
        ]))

        self._build_menus()

    # ------------------------------------------------------------------
    # 界面构建
    # ------------------------------------------------------------------
    @staticmethod
    def _domain_group(title, buttons):
        """生成一个功能域分组：标题 + 居中排列的按钮。"""
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.addStretch(1)
        for text, slot in buttons:
            row = QHBoxLayout()
            row.addStretch(1)
            btn = QPushButton(text)
            btn.setFixedSize(BUTTON_WIDTH, BUTTON_HEIGHT)
            btn.clicked.connect(slot)
            row.addWidget(btn)
            row.addStretch(1)
            layout.addLayout(row)
        layout.addStretch(1)
        return group

    def _build_menus(self):
        menubar = self.menuBar()

        menu_settings = menubar.addMenu("设置")
        menu_style = menu_settings.addMenu("界面风格")
        menu_style.addAction("Fusion", set_style_fusion)
        menu_style.addAction("Windows", set_style_windows)
        menu_style.addAction("GTK+", set_style_gtk)

        menu_font = menu_settings.addMenu("字体大小")
        menu_font.addAction("小", set_font_size_small)
        menu_font.addAction("中", set_font_size_medium)
        menu_font.addAction("大", set_font_size_large)

        menu_settings.addSeparator()
        self.action_theme = menu_settings.addAction("深色主题", self.toggle_theme)
        menu_settings.addAction("恢复默认配置文件", self.restore_default_config)

        menu_help = menubar.addMenu("帮助")
        menu_help.addAction("关于", self.show_about)

    # ------------------------------------------------------------------
    # 子窗口管理
    # ------------------------------------------------------------------
    def _show_window(self, key, factory):
        """懒加载打开子窗口；已存在则前置显示。"""
        win = self._windows.get(key)
        if win is None:
            win = factory()
            self._windows[key] = win
        win.show()
        win.raise_()
        win.activateWindow()
        return win

    def open_trajectory_page(self):
        return self._show_window('trajectory', lambda: TrajectoryPage(self.ctx, self))

    def open_datalink_page(self):
        return self._show_window('datalink', lambda: DatalinkPage(self.ctx, self))

    def open_image_gen_page(self):
        return self._show_window('image_gen', lambda: ImageGenPage(self.ctx, self.open_navi_window, self))

    def open_ins_gnss_page(self):
        return self._show_window('ins_gnss', lambda: InsGnssPage(self.ctx, self))

    def open_data_preprocess_page(self):
        return self._show_window(
            'data_preprocess',
            lambda: DataPreprocessPage(self.ctx, open_fusion=self.open_info_fusion_page, parent=self))

    def open_info_fusion_page(self):
        return self._show_window(
            'info_fusion',
            lambda: InfoFusionPage(self.ctx, open_navi=self.open_navi_window,
                                   open_post=self.open_post_process_page, parent=self))

    def open_post_process_page(self):
        return self._show_window(
            'post_process',
            lambda: PostProcessPage(self.ctx, self))

    def open_navi_window(self):
        return self._show_window('navi', lambda: NaviWindow(self.ctx, self))

    # ------------------------------------------------------------------
    # 设置 / 帮助
    # ------------------------------------------------------------------
    def toggle_theme(self):
        app = QApplication.instance()
        if self.dark_theme:
            app.setStyleSheet("")
            self.action_theme.setText("深色主题")
            self.dark_theme = False
        else:
            with open("ui/dark.qss", "r", encoding="utf-8") as f:
                app.setStyleSheet(f.read())
            self.action_theme.setText("浅色主题")
            self.dark_theme = True

    def restore_default_config(self):
        cfg_path = "configs/default.yaml"
        restore_config(cfg_path)
        self.ctx.reload_config()
        QMessageBox.information(self, "提示", f"恢复默认配置文件成功！\n{Path(cfg_path).resolve()}", QMessageBox.Ok)

    def show_about(self):
        QMessageBox.about(
            self,
            "关于",
            "<b>软件名称：</b> 无人相对导航全数字仿真验证环境<br>"
            "<b>版本号：</b> v1.2<br>"
            "<b>完成单位：</b> 西北工业大学<br>"
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei", 12))

    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    # 替换标准输出和错误输出
    sys.stdout = Logger(log_dir='logs', also_print=True)
    sys.stderr = sys.stdout

    main()
