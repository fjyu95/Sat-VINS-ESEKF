"""图像生成单元页面（对应 ui/samples/4_图像生成单元GUI界面.png）。

「解算」按钮接入真实的视觉/惯性融合导航窗口（NaviWindow），
其余参数控件可交互但不与导航核心耦合。
"""
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QDoubleSpinBox, QComboBox, QPushButton, QMessageBox, QFileDialog, QLineEdit,
)

from ui.pages.common import StorageSettingGroup, make_mode_group


class ImageGenPage(QMainWindow):
    def __init__(self, ctx=None, open_navi=None, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.open_navi = open_navi   # 由主界面注入：打开 NaviWindow
        self.setWindowTitle("图像生成单元")
        self.resize(620, 560)
        self.ins_file = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # 模式选择
        mode_group, self.mode_radios = make_mode_group(parent=self)
        root.addWidget(mode_group)

        # 惯导信息
        ins_group = QGroupBox("惯导信息")
        form = QFormLayout(ins_group)

        file_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.file_edit.setPlaceholderText("未选择文件")
        self.btn_open_ins = QPushButton("打开...")
        self.btn_open_ins.clicked.connect(self.pick_ins_file)
        file_row.addWidget(self.file_edit, stretch=1)
        file_row.addWidget(self.btn_open_ins)
        form.addRow("从文件打开：", file_row)

        self.spin_delay = QDoubleSpinBox()
        self.spin_time_err = QDoubleSpinBox()
        self.spin_height_err = QDoubleSpinBox()
        self.spin_boresight_err = QDoubleSpinBox()
        for spin in (self.spin_delay, self.spin_time_err, self.spin_height_err, self.spin_boresight_err):
            spin.setDecimals(2)
            spin.setRange(0.0, 1e6)
        form.addRow("传输时延：", self.spin_delay)
        form.addRow("时间不同步误差：", self.spin_time_err)
        form.addRow("高度测量误差：", self.spin_height_err)
        form.addRow("图像传感器视轴指向误差：", self.spin_boresight_err)

        self.combo_fault = QComboBox()
        self.combo_fault.addItems(["NoError", "图像缺失", "特征匹配失败", "位姿解算失败", "数据中断"])
        form.addRow("故障/错误信息：", self.combo_fault)
        root.addWidget(ins_group)

        # 存储设定
        self.storage_group = StorageSettingGroup(self)
        root.addWidget(self.storage_group)

        # 解算 -> 打开视觉/惯性融合导航窗口
        self.btn_solve = QPushButton("解算")
        self.btn_solve.setMinimumWidth(140)
        self.btn_solve.clicked.connect(self.solve)
        root.addWidget(self.btn_solve, 0)
        root.addStretch(1)

    def pick_ins_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开惯导信息文件", "configs", "All Files (*);;Config Files (*.yaml);;Text Files (*.txt)")
        if path:
            self.ins_file = path
            self.file_edit.setText(Path(path).name)
            self.file_edit.setToolTip(path)

    def solve(self):
        if self.open_navi is not None:
            self.open_navi()   # 进入视觉/惯性融合导航仿真界面
        else:
            QMessageBox.information(self, "解算", "导航仿真界面不可用。", QMessageBox.Ok)
