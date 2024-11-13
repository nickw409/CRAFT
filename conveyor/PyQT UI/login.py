import sys
from PySide6.QtWidgets import (QLabel, QWidget, QGridLayout, QLineEdit, QHBoxLayout, QPushButton )
from PySide6.QtCore import Qt, Signal
from firebase_utils import login

class Login(QWidget):
    login_success_signal = Signal(str)
    go_to_register = Signal()

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
        
        # buttons
        button_box = QHBoxLayout()
        login_button = QPushButton("Login")
        login_button.clicked.connect(self.handle_login)
        register_button = QPushButton("Register")
        register_button.clicked.connect(self.handle_register)
        button_box.addWidget(login_button)
        button_box.addWidget(register_button)
        grid.addLayout(button_box, 4, 1, 1, 2)

        # stretch grid around columns/rows
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(3, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(5, 1)
        
        # set the layout
        self.setLayout(grid)

    def handle_login(self):
        email = login(self.user_entry.text(), self.pass_entry.text())

        if email:
            self.login_success_signal.emit(email)
        else:
            self.error_label.setText("Invalid Credentials")
            self.error_label.setStyleSheet("font-weight: bold; color: red;")  
    
    def handle_register(self):
        self.go_to_register.emit()