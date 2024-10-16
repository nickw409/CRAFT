import sys
from PySide6.QtWidgets import (QLabel, QWidget, QGridLayout, QLineEdit, QPushButton )
from PySide6.QtCore import Qt, Signal
from firebase_utils import register

class Register(QWidget):
    register_success_signal = Signal()

    def __init__(self):
        super().__init__()
        grid = QGridLayout()

        # items
        self.error_label = QLabel("")
        grid.addWidget(self.error_label, 1, 1, 1, 2)
        self.error_label.setAlignment(Qt.AlignCenter)
        grid.addWidget(QLabel("Email: "), 2, 1)
        grid.addWidget(QLabel("Password: "), 3, 1)
        self.user_entry = QLineEdit()
        self.pass_entry = QLineEdit()
        self.pass_entry.setEchoMode(QLineEdit.Password)
        grid.addWidget(self.user_entry, 2, 2)
        grid.addWidget(self.pass_entry, 3, 2)
        
        # button
        register_button = QPushButton("Register")
        register_button.clicked.connect(self.handle_register)
        grid.addWidget(register_button, 4, 1, 1, 2)

        # stretch grid around columns/rows
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(3, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(5, 1)
        
        # setting the layout)
        self.setLayout(grid)

    def handle_register(self):
        success = register(self.user_entry.text(), self.pass_entry.text())
        if success:
            self.register_success_signal.emit()
        else:
            self.error_label.setText("Email already associated with account")
            self.error_label.setStyleSheet("font-weight: bold; color: red;") 
