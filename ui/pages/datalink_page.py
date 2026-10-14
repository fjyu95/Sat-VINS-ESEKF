"""机间链仿真单元页面（对应 ui/samples/3_轨迹生成域-机间链仿真单元GUI界面.png）。"""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QGroupBox, QFormLayout,
    QDoubleSpinBox, QPushButton, QMessageBox,
)

from ui.pages.common import StorageSettingGroup, make_mode_group


class DatalinkPage(QMainWindow):
    def __init__(self, ctx=None, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("机间链仿真单元")
        self.resize(560, 420)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # 模式选择
        mode_group, self.mode_radios = make_mode_group(parent=self)
        root.addWidget(mode_group)

        # 更改参数
        param_group = QGroupBox("更改参数")
        form = QFormLayout(param_group)
        self.spin_range_err = QDoubleSpinBox()
        self.spin_angle_err = QDoubleSpinBox()
        for spin in (self.spin_range_err, self.spin_angle_err):
            spin.setDecimals(2)
            spin.setRange(0.0, 1e6)
            spin.setMinimumWidth(200)
        form.addRow("测距误差：", self.spin_range_err)
        form.addRow("测角误差：", self.spin_angle_err)
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

    def solve(self):
        mode = self.mode_radios[0].text() if self.mode_radios[0].isChecked() else self.mode_radios[1].text()
        storage = self.storage_group.storage_dir() or "默认路径"
        summary = (f"模式：{mode}\n"
                   f"测距误差：{self.spin_range_err.value():.2f}\n"
                   f"测角误差：{self.spin_angle_err.value():.2f}\n"
                   f"存储路径：{storage}\n\n解算完成。")
        print(">>> 机间链仿真单元解算：\n" + summary)
        QMessageBox.information(self, "解算", summary, QMessageBox.Ok)
