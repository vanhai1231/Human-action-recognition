import sys
from PyQt5.QtWidgets import QApplication, QMainWindow


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Demo PyQt5'
        self.left = 10
        self.top = 10
        self.width = 320
        self.height = 200
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
