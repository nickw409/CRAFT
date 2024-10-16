import sys
from PySide6.QtWidgets import (
    QMainWindow, QApplication,
    QLabel, QToolBar, QStackedLayout, QWidget, QGridLayout, QListWidget, QVBoxLayout, QLineEdit, QHBoxLayout, QPushButton )
from PySide6.QtCore import Qt

class ImageProcess(QWidget):

    def __init__(self):
        super().__init__()
        grid = QGridLayout()

        # items
        process_label = QLabel("Image Processing Page")
        grid.addWidget(process_label)

        # setting the layout)
        self.setLayout(grid)