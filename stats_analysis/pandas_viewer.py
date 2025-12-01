import sys
import pandas as pd
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTableView,
                             QVBoxLayout, QWidget, QLabel)
from PyQt5.QtCore import (Qt, QAbstractTableModel, QModelIndex)


class PandasModel(QAbstractTableModel):
    """https://github.com/eyllanesc/QtStacking 搬运"""

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df

    # -------- 基本接口 --------
    def rowCount(self, parent=QModelIndex()):
        return self._df.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            return str(self._df.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        if orientation == Qt.Vertical:
            return str(self._df.index[section])
        return None

    # -------- 排序 --------
    def sort(self, column, order=Qt.AscendingOrder):
        col_name = self._df.columns[column]
        self.layoutAboutToBeChanged.emit()
        self._df = self._df.sort_values(col_name, ascending=(order == Qt.AscendingOrder))
        self.layoutChanged.emit()

        return None


class PandasViewer(QMainWindow):
    def __init__(self, parent=None, df=None, path='registration_stats.csv'):
        super().__init__(parent)
        self.setWindowTitle("视觉导航图像配准结果统计")
        self.resize(900, 800)

        # 1. 造点数据
        if df is None:
            df = pd.read_csv(path, sep=' ')

        # 2. 模型 + 视图
        self.model = PandasModel(df)
        self.view = QTableView()
        self.view.setModel(self.model)
        self.view.setSortingEnabled(True)  # 点击表头排序
        self.view.setAlternatingRowColors(True)
        self.view.resizeColumnsToContents()

        # 3. 状态栏显示行列数
        self.statusBar().showMessage(f"Rows: {df.shape[0]}  Columns: {df.shape[1]}")

        # 4. 布局
        central = QWidget()
        layout = QVBoxLayout(central)
        # layout.addWidget(QLabel("图像配准结果统计："))
        layout.addWidget(self.view)
        self.setCentralWidget(central)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PandasViewer()
    win.show()
    sys.exit(app.exec_())
