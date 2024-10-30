import sys
from PySide6.QtWidgets import ( QMainWindow, QApplication,
QLabel, QStackedLayout, QWidget, QVBoxLayout)
from PySide6.QtCore import Qt
from login import Login
from register import Register
from image_process import ImageProcess

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Sherd Conveyor System')
        self.setFixedSize(900, 500)
        layout = QVBoxLayout()
        self.pages = QStackedLayout()

        login = Login()
        login.login_success_signal.connect(self.show_process_page)
        login.go_to_register.connect(self.show_register_page)
        register = Register()
        register.register_success_signal.connect(self.show_login_page)       

        # add pages
        self.pages.addWidget(login)
        self.pages.addWidget(ImageProcess())
        self.pages.addWidget(register)

        # REMOVE THIS TO ENABLE LOGIN AND REGISTER
        self.pages.setCurrentIndex(1)

        # craft banner
        title_label = QLabel("CRAFT")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font: 40px; font-weight: bold; color: dark brown; border: 10px solid dark brown")  
        layout.addWidget(title_label) 

        # setting the layout
        stackedLayoutWidget = QWidget()
        stackedLayoutWidget.setLayout(self.pages)
        layout.addWidget(stackedLayoutWidget)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def show_process_page(self):
        self.pages.setCurrentIndex(1)

    def show_login_page(self):
        self.pages.setCurrentIndex(0)

    def show_register_page(self):
        self.pages.setCurrentIndex(2)
        

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()