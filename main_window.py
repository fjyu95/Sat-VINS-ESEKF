import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, \
    QGraphicsEllipseItem, QFileDialog, QMessageBox, QPushButton, QGraphicsTextItem, QProgressBar, QLabel, QVBoxLayout, \
    QProgressDialog, QAction, QToolBar, QSizePolicy
from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QPainterPath, QPen, QColor, QPainter, QBrush, QFont, QFontMetrics, QPalette, \
    QKeySequence
from PyQt5 import uic

from utils.config import restore_config
from utils.core_utils import Logger
from stats_analysis.image_viewer import ImageBrowser
from stats_analysis.navi_stats_plot import plot_bar_line_with_stats
from stats_analysis.pandas_viewer import PandasViewer
from utils.parser import load_config
from preprocessing.dehaze.dehaze import dehaze_image
from preprocessing.enhance.enhance import clahe_enhance
from vio import EKFWorker
from utils.viz import draw_traj

IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', ".webp", '.tif', '.tiff']


class PreprocessThread(QThread):
    progress = pyqtSignal(int)  # 发射进度（0~100）
    finished = pyqtSignal(str, str)

    def __init__(self, files, output_dir, func, func_name, **kwargs):
        super().__init__()
        self.files = files
        self.output_dir = output_dir
        self.func = func
        self.func_name = func_name
        self.kwargs = kwargs

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)

        n_files = len(self.files)
        for i, path in enumerate(self.files, start=1):
            try:
                data_hazy, clean_image = self.func(path, **self.kwargs)
                save_path = os.path.join(self.output_dir, f'{self.func_name}_{Path(path).name}')
                cv2.imwrite(save_path, clean_image)
                print(f'{i} save {self.func_name} image to {save_path}')

                save_path = os.path.join(self.output_dir, f'combined_{Path(path).name}')
                combined = np.hstack((data_hazy, clean_image))
                cv2.imwrite(save_path, combined)

                # time.sleep(1)
                progress = int(i / n_files * 100)
                self.progress.emit(progress)
            except Exception as e:
                print(e)
                break

        self.finished.emit(self.func_name, self.output_dir)


class ZoomableGraphicsView(QGraphicsView):
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


def set_style_Fusion():
    QApplication.setStyle("Fusion")


def set_style_Windows():
    QApplication.setStyle("Windows")


def set_style_GTKplus():
    QApplication.setStyle("GTK+")


def set_font_size_small():
    QApplication.setFont(QFont("Microsoft YaHei", 12))


def set_font_size_medium():
    QApplication.setFont(QFont("Microsoft YaHei", 14))


def set_font_size_large():
    QApplication.setFont(QFont("Microsoft YaHei", 16))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.progress_dialog = None
        uic.loadUi('ui/main_window.ui', self)

        # self.setWindowTitle("GeoVIO")
        self.resize(1800, 1000)
        self.background = None

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

        self.actionFusion.triggered.connect(set_style_Fusion)
        self.actionWindows.triggered.connect(set_style_Windows)
        self.actionGTK.triggered.connect(set_style_GTKplus)

        self.action_font_small.triggered.connect(set_font_size_small)
        self.action_font_medium.triggered.connect(set_font_size_medium)
        self.action_font_large.triggered.connect(set_font_size_large)

        self.dark_theme = False
        self.action_bg_mode.triggered.connect(self.change_theme)

        self.action_restore_config.triggered.connect(self.restore_default_config)

        self.cur_plane_position = None
        self.plane_label_position = None

        self.basemap_path = None
        self.config = load_config()  # 加载默认配置文件
        self.auto_load_basemap()

        self.dehazed = None
        self.enhanced = None

        self.ekf_running = False  # 是否已经 start() 过
        self.ekf_paused = False  # 当前是否处于暂停状态

        self.action_about.triggered.connect(self.btn_show_about)
        self.action_load_config.triggered.connect(self.btn_load_config)

        self.action_dehaze.triggered.connect(self.btn_dehaze)
        self.action_enhance.triggered.connect(self.btn_enhance)

        self.action_simulate.triggered.connect(self.btn_start_navigation)
        # self.action_pause.triggered.connect(self.btn_end_navigation)

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

        # self.btn_start_navigation()  # auto start navigation,for debug

    def display_matches(self):
        try:
            self.image_viewer = ImageBrowser(self)
            self.image_viewer.open_folder('image_matches_results')
            self.image_viewer.show()
        except Exception as e:
            print(e)
            QMessageBox.critical(self, "错误", f"查看图像匹配结果失败！\n{e}", QMessageBox.Ok)

    def display_pandas(self):
        try:
            self.pandas_viewer = PandasViewer(self, path='results/registration_stats.csv')
            self.pandas_viewer.show()
        except Exception as e:
            print(e)
            QMessageBox.critical(self, "错误", f"查看图像配准统计数据失败！\n{e}", QMessageBox.Ok)

    def display_evo_traj(self):
        try:
            draw_traj('results/traj_esekf.csv', 'results/traj_gt.csv')
        except Exception as e:
            print(e)
            QMessageBox.critical(self, "错误", f"显示导航轨迹失败！\n{e}", QMessageBox.Ok)

    def display_ekf_stats(self):
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
        except Exception as e:
            print(e)
            QMessageBox.critical(self, "错误", f"显示视觉导航误差统计失败！\n{e}", QMessageBox.Ok)

    def load_qss(self, path):
        with open(path, "r", encoding="utf-8") as f:
            qss = f.read()
            self.setStyleSheet(qss)

    def change_theme(self):
        if self.dark_theme:
            # self.apply_light_theme()
            # 清空样式文件
            self.setStyleSheet("")
            # 恢复系统默认调色板
            self.setPalette(self.style().standardPalette())

            self.dark_theme = False
            self.action_bg_mode.setText("深色主题")
        else:
            self.apply_dark_theme()
            self.dark_theme = True
            self.action_bg_mode.setText("浅色主题")

    def apply_dark_theme(self):
        self.load_qss("ui/dark.qss")
        self.dark_theme = True

    def apply_light_theme(self):
        self.load_qss("ui/light.qss")

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
                self.config = config
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
        QMessageBox.information(self, "提示", f"恢复默认配置文件成功！\n{Path(cfg_path).resolve()}", QMessageBox.Ok)

    # 只能对原图去雾，不能先增强再去雾
    def btn_dehaze(self):
        images_dir = Path(self.config['aerial_images_dir'])
        if self.config['DEBUG']:
            images_dir = Path('preprocessing/dehaze/test_data')  # for debug
        output_dir = images_dir.parent.resolve() / 'dehazed'

        image_files = [str(p) for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        sorted_files = sorted(image_files)

        try:
            img = cv2.imread(sorted_files[0], -1)
            if img.ndim == 2:
                QMessageBox.information(self, "操作无效", f"暂不支持灰度图像去雾！\n{images_dir}", QMessageBox.Ok)
                return
        except Exception as e:
            print(f'Exception {images_dir}, {e}')
            return

        if self.progress_dialog is None:
            # 创建弹窗进度条
            self.progress_dialog = QProgressDialog("正在处理……", "取消", 0, 100, self)
            self.progress_dialog.setWindowTitle("任务进度")
            self.progress_dialog.setAutoClose(True)  # 100% 自动关闭
            self.progress_dialog.setAutoReset(True)
            self.progress_dialog.setModal(True)
            # self.progress_dialog.show()

        self.worker = PreprocessThread(sorted_files, output_dir.as_posix(), dehaze_image, 'dehazed')
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.handle_results)
        self.worker.start()

    def btn_enhance(self):
        if self.dehazed:
            images_dir = Path(self.dehazed)
        else:
            # images_dir = Path('preprocessing/enhance/test_data')
            images_dir = Path(self.config['aerial_images_dir'])
        output_dir = images_dir.parent.resolve() / 'enhanced'

        image_files = [str(p) for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if self.dehazed:
            # 只增强去雾过的图像，过滤combined
            image_files = [p for p in image_files if Path(p).name.startswith('dehazed_')]
        sorted_files = sorted(image_files)

        if self.progress_dialog is None:
            # 创建弹窗进度条
            self.progress_dialog = QProgressDialog("正在处理……", "取消", 0, 100, self)
            self.progress_dialog.setWindowTitle("任务进度")
            self.progress_dialog.setAutoClose(True)  # 100% 自动关闭
            self.progress_dialog.setAutoReset(True)
            self.progress_dialog.setModal(True)
            # self.progress_dialog.show()

        kwargs = {'clip_limit': self.config['clahe_clip_limit']}
        self.worker = PreprocessThread(sorted_files, output_dir.as_posix(), clahe_enhance, 'enhanced', **kwargs)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.handle_results)
        self.worker.start()

    def update_progress(self, value):
        self.progress_dialog.setValue(value)

    def handle_results(self, func_name, output_dir):
        if func_name == 'dehazed':
            self.dehazed = output_dir
            QMessageBox.information(self, "处理成功", f"图像去雾结果保存在 {output_dir}", QMessageBox.Ok)
        elif func_name == 'enhanced':
            self.enhanced = output_dir
            QMessageBox.information(self, "处理成功", f"图像增强结果保存在 {output_dir}", QMessageBox.Ok)
        else:
            print(f'Unknown function name: {func_name}')

        print(f'>>> newest images switch to {output_dir}')

    def set_final_images(self):
        # self.enhanced = 'dataset/enhanced'  # for debug
        if self.enhanced:
            preprocessed_images = [p for p in Path(self.enhanced).iterdir() if
                                   p.suffix.lower() in IMAGE_EXTS and p.name.startswith('enhanced_')]
        elif self.dehazed:
            preprocessed_images = [p for p in Path(self.dehazed).iterdir() if
                                   p.suffix.lower() in IMAGE_EXTS and p.name.startswith('dehazed_')]
        else:
            self.config['final_files'] = None
            return

        self.config['final_files'] = sorted(preprocessed_images)

    def btn_start_navigation(self):
        if not self.ekf_running:
            # --- 线程和 EKF 设置 ---
            self.thread = QThread()  # 1. 创建 QThread 实例
            self.set_final_images()
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
                # self.action_stop.setEnabled(False)  # 暂停状态下不允许停止
            else:
                # 当前是暂停 -> 切换为运行
                self.ekf_worker.pause(False)
                self.ekf_paused = False
                self.action_start.setText("暂停")
                self.action_simulate.setText("暂停导航")
                # self.action_stop.setEnabled(True)

    def update_ekf_label(self, i, n):
        ekf_label = f"EKF 任务进度: {i}/{n}"
        if i == n:
            ekf_label += " | 状态: 完成"

        self.ekf_label.setText(ekf_label)
        self.ekf_label.adjustSize()

    def end_navigation(self):
        print(">>> EKF 线程结束,重置导航状态")
        # self.ekf_label.setText("EKF 任务进度: 0% | 状态: 待机")
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
            "<b>完成单位：</b> xxx<br>"
        )

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
        # print(f'font distance: {distance}, w: {w}, h: {h}')

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
        # text.setOpacity(0.7)
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
        # scene.clear()

        # 清空 scene 但保留底图
        for item in self.scene.items():
            if item is not self.background:  # 保留底图
                self.scene.removeItem(item)


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
