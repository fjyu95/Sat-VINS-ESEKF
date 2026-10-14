"""惯性/卫星导航仿真单元页面（对应 ui/samples/5_惯性卫星导航仿真单元GUI界面.png）。"""
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QDoubleSpinBox, QPushButton, QMessageBox, QFileDialog, QLineEdit,
)

from ui.pages.common import StorageSettingGroup, make_mode_group


class InsGnssPage(QMainWindow):
    def __init__(self, ctx=None, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("惯性/卫星导航仿真单元")
        self.resize(620, 500)
        self.param_file = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # 模式选择
        mode_group, self.mode_radios = make_mode_group(parent=self)
        root.addWidget(mode_group)

        # 参数信息
        param_group = QGroupBox("参数信息")
        form = QFormLayout(param_group)

        file_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.file_edit.setPlaceholderText("未选择文件")
        self.btn_open_file = QPushButton("打开...")
        self.btn_open_file.clicked.connect(self.pick_param_file)
        file_row.addWidget(self.file_edit, stretch=1)
        file_row.addWidget(self.btn_open_file)
        form.addRow("从文件打开：", file_row)

        self.spin_install_err = QDoubleSpinBox()
        self.spin_scale_err = QDoubleSpinBox()
        self.spin_gyro_drift = QDoubleSpinBox()
        self.spin_acc_bias = QDoubleSpinBox()
        for spin in (self.spin_install_err, self.spin_scale_err, self.spin_gyro_drift, self.spin_acc_bias):
            spin.setDecimals(2)
            spin.setRange(-1e6, 1e6)
        form.addRow("安装误差：", self.spin_install_err)
        form.addRow("刻度字数误差：", self.spin_scale_err)
        form.addRow("陀螺漂移：", self.spin_gyro_drift)
        form.addRow("加速度计零位：", self.spin_acc_bias)
        root.addWidget(param_group)

        # 存储设定
        self.storage_group = StorageSettingGroup(self)
        root.addWidget(self.storage_group)

        # 解算
        self.btn_solve = QPushButton("解算")
        self.btn_solve.setMinimumWidth(140)
        self.btn_solve.clicked.connect(self.solve)
        root.addWidget(self.btn_solve, 0)
        root.addStretch(1)

    def pick_param_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开参数文件", "configs", "All Files (*);;Config Files (*.yaml);;Text Files (*.txt)")
        if path:
            self.param_file = path
            self.file_edit.setText(Path(path).name)
            self.file_edit.setToolTip(path)

    def solve(self):
        mode = self.mode_radios[0].text() if self.mode_radios[0].isChecked() else self.mode_radios[1].text()
        storage = self.storage_group.storage_dir() or "默认路径"
        summary = (f"模式：{mode}\n"
                   f"安装误差：{self.spin_install_err.value():.2f}\n"
                   f"刻度字数误差：{self.spin_scale_err.value():.2f}\n"
                   f"陀螺漂移：{self.spin_gyro_drift.value():.2f}\n"
                   f"加速度计零位：{self.spin_acc_bias.value():.2f}\n"
                   f"参数文件：{self.param_file or '无'}\n"
                   f"存储路径：{storage}\n\n解算完成。")
        print(">>> 惯性/卫星导航仿真单元解算：\n" + summary)
        QMessageBox.information(self, "解算", summary, QMessageBox.Ok)
