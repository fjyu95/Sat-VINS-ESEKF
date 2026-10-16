"""图像/视觉导航仿真单元主界面（底图 + EKF 导航可视化）。

由原 ``main_window.py`` 中的 ``MainWindow`` 迁移而来，核心交互与流程保持不变：
加载配置 -> （可选）图像去雾/增强 -> 开始导航 -> 统计分析查看。
差异仅在于配置与预处理结果改由 ``AppContext`` 共享，供数据处理层页面协同调用。
"""
from pathlib import Path

import cv2
import numpy as np

from PyQt5.QtWidgets import QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, \
    QFileDialog, QMessageBox, QGraphicsTextItem, QLabel
from PyQt5.QtCore import Qt, QTimer, QThread
from PyQt5.QtGui import QPixmap, QPen, QColor, QPainter, QBrush, QFont
from PyQt5 import uic

from utils.config import restore_config
from utils.parser import load_config
from preprocessing.dehaze.dehaze import dehaze_image
from preprocessing.enhance.enhance import clahe_enhance
from vio import EKFWorker
from ui.pages.context import IMAGE_EXTS
from ui.pages.common import (
    ZoomableGraphicsView,
    set_style_fusion, set_style_windows, set_style_gtk,
    set_font_size_small, set_font_size_medium, set_font_size_large,
    show_match_results, show_registration_stats, show_est_vs_gt_traj, compute_ekf_error_stats,
)
from ui.pages.workers import run_preprocess


class NaviWindow(QMainWindow):
    """视觉/惯性融合导航仿真窗口。"""

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        uic.loadUi('ui/main_window.ui', self)
        self.setWindowTitle("无人相对导航全数字仿真验证环境")

        self.resize(1800, 1000)
        self.background = None
        self._stat_viewers = []   # 统计子窗口引用，防止被 GC

        # 替换 Designer 的 mapView 为可缩放版本
        old_view = self.findChild(QGraphicsView, "mapView")
        self.view = ZoomableGraphicsView(old_view.parent())
        self.view.setGeometry(old_view.geometry())
        old_view.hide()  # 隐藏原控件
        self.mapView = self.view  # 保留名称与 UI 一致

        # 把 view 设为中心部件
        self.setCentralWidget(self.mapView)

        # 创建场景
        self.scene = QGraphicsScene(self)
        self.mapView.setScene(self.scene)

        self.current_view_item = QGraphicsPixmapItem()
        self.scene.addItem(self.current_view_item)

        self.current_view_item.setZValue(10)
        self.current_view_item.setOpacity(0.7)

        self.mapView.setRenderHints(
            QPainter.Antialiasing |
            QPainter.SmoothPixmapTransform
        )

        # 创建 Overlay 标签（显示 EKF 进度）
        self.ekf_label = QLabel("EKF 任务进度: 0% | 状态: 待机", self.mapView.viewport())
        self.ekf_label.setStyleSheet("""
            QLabel {
                color: yellow;
                background-color: rgba(0, 0, 0, 100);
                font-family: "Microsoft YaHei";
                font-size: 20px;
                font-weight: bold;
                padding: 4px;
                border-radius: 4px;
            }
        """)
        self.ekf_label.move(5, 5)  # 左上角位置
        self.ekf_label.adjustSize()
        self.ekf_label.show()

        self.actionFusion.triggered.connect(set_style_fusion)
        self.actionWindows.triggered.connect(set_style_windows)
        self.actionGTK.triggered.connect(set_style_gtk)

        self.action_font_small.triggered.connect(set_font_size_small)
        self.action_font_medium.triggered.connect(set_font_size_medium)
        self.action_font_large.triggered.connect(set_font_size_large)

        self.dark_theme = False
        self.action_bg_mode.triggered.connect(self.change_theme)

        self.action_restore_config.triggered.connect(self.restore_default_config)

        self.cur_plane_position = None
        self.plane_label_position = None

        self.basemap_path = None
        self.auto_load_basemap()

        self.ekf_running = False  # 是否已经 start() 过
        self.ekf_paused = False   # 当前是否处于暂停状态

        self.action_about.triggered.connect(self.btn_show_about)
        self.action_load_config.triggered.connect(self.btn_load_config)

        self.action_dehaze.triggered.connect(self.btn_dehaze)
        self.action_enhance.triggered.connect(self.btn_enhance)

        self.action_simulate.triggered.connect(self.btn_start_navigation)

        self.action_clear.triggered.connect(self.clear_scene)
        self.action_resize.triggered.connect(self.initial_fit)

        self.action_start.triggered.connect(self.btn_start)
        self.action_stop.triggered.connect(self.btn_stop)
        self.action_stop.setEnabled(False)

        # 统计分析相关
        self.action_display_matches.triggered.connect(self.display_matches)
        self.action_stats_register.triggered.connect(self.display_pandas)
        self.action_evo_display.triggered.connect(self.display_evo_traj)
        self.action_stats_ekf.triggered.connect(self.display_ekf_stats)

    # ------------------------------------------------------------------
    # 共享配置
    # ------------------------------------------------------------------
    @property
    def config(self):
        return self.ctx.config

    # ------------------------------------------------------------------
    # 统计分析（委托公共辅助函数）
    # ------------------------------------------------------------------
    def display_matches(self):
        viewer = show_match_results(self)
        if viewer is not None:
            self._stat_viewers.append(viewer)

    def display_pandas(self):
        viewer = show_registration_stats(self)
        if viewer is not None:
            self._stat_viewers.append(viewer)

    def display_evo_traj(self):
        show_est_vs_gt_traj(self)

    def display_ekf_stats(self):
        compute_ekf_error_stats(self)

    # ------------------------------------------------------------------
    # 主题 / 配置
    # ------------------------------------------------------------------
    def load_qss(self, path):
        with open(path, "r", encoding="utf-8") as f:
            qss = f.read()
            self.setStyleSheet(qss)

    def change_theme(self):
        if self.dark_theme:
            # 清空样式文件
            self.setStyleSheet("")
            # 恢复系统默认调色板
            self.setPalette(self.style().standardPalette())

            self.dark_theme = False
            self.action_bg_mode.setText("深色主题")
        else:
            self.load_qss("ui/dark.qss")
            self.dark_theme = True
            self.action_bg_mode.setText("浅色主题")

    def resizeEvent(self, event):
        # 保持场景内容适应视图大小，保持宽高比
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)
        self.ekf_label.move(5, 5)  # 始终左上角

    def initial_fit(self):
        # 此时 mapView 已经拿到最大化后的真实尺寸
        self.mapView.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def show_basemap(self, path):
        self.background = self.scene.addPixmap(QPixmap(path))
        self.background.setZValue(0)
        QTimer.singleShot(0, self.initial_fit)

    def auto_load_basemap(self):
        if self.config:
            self.basemap_path = self.config['basemap_path']
            self.show_basemap(self.basemap_path)

    def btn_load_config(self):
        default_config_dir = Path('configs').resolve().as_posix()
        path, _ = QFileDialog.getOpenFileName(self, "打开文件", default_config_dir, "Config Files (*.yaml)")
        try:
            config = load_config(path)
            if config:
                self.ctx.config = config
                basemap_path = self.config['basemap_path']
                if basemap_path != self.basemap_path:
                    self.show_basemap(basemap_path)  # 避免重复加载
                    self.basemap_path = basemap_path
                QMessageBox.information(self, f"加载配置文件成功!", path, QMessageBox.Ok)
        except Exception as e:
            print("Error loading config file: ", path, e)
            QMessageBox.critical(self, "错误", f"加载配置文件失败！请检查\n{path}", QMessageBox.Ok)

    def restore_default_config(self):
        cfg_path = "configs/default.yaml"
        restore_config(cfg_path)
        self.ctx.reload_config()
        QMessageBox.information(self, "提示", f"恢复默认配置文件成功！\n{Path(cfg_path).resolve()}", QMessageBox.Ok)

    # ------------------------------------------------------------------
    # 图像预处理（去雾 / 增强）
    # ------------------------------------------------------------------
    @staticmethod
    def _list_images(images_dir, prefix=None):
        files = [str(p) for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if prefix:
            files = [p for p in files if Path(p).name.startswith(prefix)]
        return sorted(files)

    # 只能对原图去雾，不能先增强再去雾
    def btn_dehaze(self):
        images_dir = Path(self.config['aerial_images_dir'])
        if self.config['DEBUG']:
            images_dir = Path('preprocessing/dehaze/test_data')  # for debug
        output_dir = images_dir.parent.resolve() / 'dehazed'

        sorted_files = self._list_images(images_dir)
        if not sorted_files:
            QMessageBox.warning(self, "提示", f"未在目录中找到图像：\n{images_dir}", QMessageBox.Ok)
            return

        try:
            img = cv2.imread(sorted_files[0], -1)
            if img.ndim == 2:
                QMessageBox.information(self, "操作无效", f"暂不支持灰度图像去雾！\n{images_dir}", QMessageBox.Ok)
                return
        except Exception as e:
            print(f'Exception {images_dir}, {e}')
            return

        self._worker = run_preprocess(self, sorted_files, output_dir.as_posix(),
                                      dehaze_image, 'dehazed', on_finished=self.handle_results)

    def btn_enhance(self):
        if self.ctx.dehazed_dir:
            images_dir = Path(self.ctx.dehazed_dir)
        else:
            images_dir = Path(self.config['aerial_images_dir'])
        output_dir = images_dir.parent.resolve() / 'enhanced'

        # 若已有去雾结果，只增强去雾过的图像，过滤 combined
        prefix = 'dehazed_' if self.ctx.dehazed_dir else None
        sorted_files = self._list_images(images_dir, prefix=prefix)
        if not sorted_files:
            QMessageBox.warning(self, "提示", f"未在目录中找到图像：\n{images_dir}", QMessageBox.Ok)
            return

        kwargs = {'clip_limit': self.config['clahe_clip_limit']}
        self._worker = run_preprocess(self, sorted_files, output_dir.as_posix(),
                                      clahe_enhance, 'enhanced',
                                      on_finished=self.handle_results, **kwargs)

    def handle_results(self, func_name, output_dir):
        if func_name == 'dehazed':
            self.ctx.dehazed_dir = output_dir
            QMessageBox.information(self, "处理成功", f"图像去雾结果保存在 {output_dir}", QMessageBox.Ok)
        elif func_name == 'enhanced':
            self.ctx.enhanced_dir = output_dir
            QMessageBox.information(self, "处理成功", f"图像增强结果保存在 {output_dir}", QMessageBox.Ok)
        else:
            print(f'Unknown function name: {func_name}')

        print(f'>>> newest images switch to {output_dir}')

    # ------------------------------------------------------------------
    # EKF 导航线程
    # ------------------------------------------------------------------
    def btn_start_navigation(self):
        if not self.ekf_running:
            # --- 线程和 EKF 设置 ---
            self.thread = QThread()  # 1. 创建 QThread 实例
            self.ctx.set_final_images()
            self.ekf_worker = EKFWorker(self.config)  # 2. 创建 Worker 实例
            self.ekf_worker.moveToThread(self.thread)  # 3. 将 Worker 移入子线程

            # 4. 连接信号和槽：当子线程发出信号时，连接到主线程的绘图槽函数
            self.ekf_worker.plane_position_signal.connect(self.update_point)
            self.ekf_worker.draw_matches_signal.connect(self.update_registered_image)
            self.ekf_worker.pnp_line_signal.connect(self.draw_connection_line)
            self.ekf_worker.ekf_progress_signal.connect(self.update_ekf_label)
            self.ekf_worker.ekf_worker_finished_signal.connect(self.end_navigation)

            # 5. 连接线程启动：当线程启动时，执行 Worker 的 run 方法
            self.thread.started.connect(self.ekf_worker.pipeline)

            # 6. 启动 QThread (线程等待 Worker 启动命令)
            self.thread.start()

            self.ekf_running = True
            self.ekf_paused = False
            self.action_start.setText("暂停")  # 变成暂停按钮
            self.action_simulate.setText("暂停导航")
            self.action_stop.setEnabled(True)
            self.clear_scene()
            print(">>> 开始 EKF 线程")
        else:
            if not self.ekf_paused:
                # 当前在运行 -> 切换为暂停
                self.ekf_worker.pause(True)
                self.ekf_paused = True
                self.action_start.setText("继续")
                self.action_simulate.setText("继续导航")
            else:
                # 当前是暂停 -> 切换为运行
                self.ekf_worker.pause(False)
                self.ekf_paused = False
                self.action_start.setText("暂停")
                self.action_simulate.setText("暂停导航")

    def update_ekf_label(self, i, n):
        ekf_label = f"EKF 任务进度: {i}/{n}"
        if i == n:
            ekf_label += " | 状态: 完成"

        self.ekf_label.setText(ekf_label)
        self.ekf_label.adjustSize()

    def end_navigation(self):
        print(">>> EKF 线程结束,重置导航状态")
        self.btn_stop()

    def btn_start(self):
        self.btn_start_navigation()

    def btn_stop(self):
        # 停止线程和 Worker
        self.ekf_worker.stop()
        self.action_stop.setEnabled(False)

        self.thread.quit()
        self.thread.wait()

        self.action_start.setText("开始")
        self.action_simulate.setText("开始导航")
        self.action_start.setEnabled(True)
        self.ekf_running = False

    def btn_show_about(self):
        QMessageBox.about(
            self,
            "关于",
            "<b>软件名称：</b> 无人相对导航全数字仿真系统<br>"
            "<b>版本号：</b> v1.2<br>"
            "<b>完成单位：</b> 西北工业大学<br>"
        )

    # ------------------------------------------------------------------
    # 场景绘制
    # ------------------------------------------------------------------
    def update_point(self, x, y, idx):
        r = 3
        pt = self.scene.addEllipse(x - r, y - r, 2 * r, 2 * r, pen=QPen(Qt.NoPen), brush=QBrush(Qt.red))  # 画点，飞机位置
        pt.setZValue(999)

        distance = -1
        if self.cur_plane_position:
            pen = QPen(QColor(0, 255, 0), r, Qt.SolidLine)
            self.scene.addLine(self.cur_plane_position[0], self.cur_plane_position[1], x, y, pen)  # 画飞行轨迹

            delta = x - self.plane_label_position[0], y - self.plane_label_position[1]
            distance = np.linalg.norm(np.array(delta))  # 不是第一个标注位置时，计算与上一个标注的距离,避免遮挡

        font = QFont("Arial", 12, QFont.Bold)
        tmp_item = QGraphicsTextItem()
        tmp_item.setFont(font)
        tmp_item.setPlainText('')
        rect = tmp_item.boundingRect()
        w, h = rect.width(), rect.height()

        if self.plane_label_position is None or distance > max(w, h) // 2:
            text = self.scene.addText(str(idx), font)
            text.setZValue(1000)
            text.setDefaultTextColor(QColor(255, 255, 0))
            text.setPos(x + w // 2, y - h // 2)
            self.plane_label_position = (x, y)

        self.cur_plane_position = (x, y)

    def stamp_keyframe(self, pixmap, x=0, y=0, cx=0, cy=0):
        stamp = QGraphicsPixmapItem(pixmap)
        stamp.setPos(x, y)
        stamp.setZValue(1)  # 比底图高，但比当前视图低
        stamp.setOpacity(0.5)  # 历史痕迹淡一点
        self.scene.addItem(stamp)

    def update_registered_image(self, q_image, x, y, w, h, n_ekf):
        cx, cy = x + w / 2, y + h / 2

        # 1. 更新图像内容
        pixmap = QPixmap.fromImage(q_image)
        self.current_view_item.setPixmap(pixmap)
        self.current_view_item.setPos(x, y)
        self.stamp_keyframe(pixmap, x, y, cx, cy)

        font = QFont("Arial", 24, QFont.Bold)
        text = self.scene.addText(str(n_ekf), font)
        text.setDefaultTextColor(QColor(255, 255, 0))
        text.setZValue(1000)
        rect = text.boundingRect()

        dx = rect.width() // 2
        dy = rect.height() // 2
        text.setPos(cx - dx, cy - dy)

    def draw_connection_line(self, x1, y1, x2, y2, idx):
        r = 3
        pen = QPen(QColor(0, 255, 0), r, Qt.SolidLine)
        self.scene.addLine(x1, y1, x2, y2, pen)

    def clear_scene(self):
        # 清空 scene 但保留底图
        for item in self.scene.items():
            if item is not self.background:  # 保留底图
                self.scene.removeItem(item)
