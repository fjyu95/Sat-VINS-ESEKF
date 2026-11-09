"""功能域-故障模拟单元页面（简化版）。

选择文本类数据文件（航拍元数据 txt / 轨迹 csv 等），注入三种简单故障：
数值偏置 / 数值噪声 / 数据中断。

注入方式：选择文件 → 原文件备份为 *.bak → 故障内容覆盖写回原路径。
对航拍图像同名 txt 元数据注入后，导航仿真（EKFWorker）会真实读到故障数据；
「还原原文件」从备份恢复。
"""
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QComboBox, QDoubleSpinBox, QPushButton, QLineEdit, QFileDialog,
    QMessageBox, QLabel,
)

from preprocessing.fault_inject import inject_file, restore_file

FAULT_KINDS = [('数值偏置', 'bias'), ('数值噪声', 'noise'), ('数据中断', 'drop')]


class FaultSimPage(QMainWindow):
    """故障模拟单元：选择文件注入故障/错误数据。"""

    def __init__(self, ctx=None, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("故障模拟单元")
        self.resize(560, 320)
        self.fault_file = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        group = QGroupBox("故障注入")
        form = QFormLayout(group)

        self.combo_kind = QComboBox()
        self.combo_kind.addItems([name for name, _ in FAULT_KINDS])
        form.addRow("故障类型：", self.combo_kind)

        self.spin_strength = QDoubleSpinBox()
        self.spin_strength.setRange(0.0, 100.0)
        self.spin_strength.setValue(20.0)
        self.spin_strength.setSuffix(" %")
        form.addRow("故障强度：", self.spin_strength)

        file_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.file_edit.setPlaceholderText("未选择文件（支持 txt/csv 等数值文本）")
        self.btn_pick = QPushButton("选择文件…")
        self.btn_pick.clicked.connect(self.pick_file)
        file_row.addWidget(self.file_edit, 1)
        file_row.addWidget(self.btn_pick)
        form.addRow("注入文件：", file_row)

        hint = QLabel("注入后原文件备份为 *.bak，故障覆盖写回原路径；"
                      "对航拍元数据 txt 注入后导航仿真直接生效。")
        hint.setWordWrap(True)
        form.addRow("", hint)
        root.addWidget(group)

        op_row = QHBoxLayout()
        self.btn_inject = QPushButton("注入故障")
        self.btn_restore = QPushButton("还原原文件")
        self.btn_inject.setMinimumHeight(32)
        self.btn_restore.setMinimumHeight(32)
        self.btn_inject.clicked.connect(self.inject)
        self.btn_restore.clicked.connect(self.restore)
        op_row.addWidget(self.btn_inject)
        op_row.addWidget(self.btn_restore)
        op_row.addStretch(1)
        root.addLayout(op_row)

        self.status_label = QLabel("状态：待机")
        root.addWidget(self.status_label)
        root.addStretch(1)

    def _set_status(self, text):
        self.status_label.setText(f"状态：{text}")
        print(f">>> {text}")

    def pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择要注入的数据文件", ".",
            "Data Files (*.txt *.csv);;All Files (*)")
        if path:
            self.fault_file = path
            self.file_edit.setText(Path(path).name)
            self.file_edit.setToolTip(path)

    def inject(self):
        if not self.fault_file:
            QMessageBox.information(self, "提示", "请先选择要注入的文件。", QMessageBox.Ok)
            return
        kind = dict(FAULT_KINDS)[self.combo_kind.currentText()]
        try:
            n, bak = inject_file(self.fault_file, kind, self.spin_strength.value())
        except Exception as e:
            print(e)
            QMessageBox.critical(self, "错误", f"故障注入失败！\n{e}", QMessageBox.Ok)
            return

        msg = (f"故障类型：{self.combo_kind.currentText()}\n"
               f"故障强度：{self.spin_strength.value():.0f}%\n"
               f"修改数值：{n} 处\n"
               f"目标文件：{self.fault_file}\n"
               f"原文件备份：{bak}")
        self._set_status(f"故障已注入（{n} 处数值）→ {self.fault_file}")
        QMessageBox.information(self, "注入成功", msg, QMessageBox.Ok)

    def restore(self):
        if not self.fault_file:
            QMessageBox.information(self, "提示", "请先选择文件。", QMessageBox.Ok)
            return
        if restore_file(self.fault_file):
            self._set_status(f"已从备份还原：{self.fault_file}")
            QMessageBox.information(self, "还原", "原文件已恢复，故障注入失效。", QMessageBox.Ok)
        else:
            QMessageBox.information(self, "提示", "未找到备份文件（*.bak），无需还原。",
                                    QMessageBox.Ok)
