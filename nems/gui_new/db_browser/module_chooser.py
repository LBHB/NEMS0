import sys

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
import PyQt5.QtWidgets as qw
import PyQt5.QtCore as qc
import PyQt5.QtGui as qg


class ChooserWindow(qw.QMainWindow):

    def __init__(self, modelspec=None, xfspec=None, rec=None, ctx=None,
                 rec_name='val', control_widget=None):
        '''
        Main Window wrapper
        '''
        super(qw.QMainWindow, self).__init__()
        self.title = 'Module Chooser'

        self.setCentralWidget(module_chooser())
        self.setWindowTitle(self.title)
        self.show()

# Subclass QMainWindow to customize your application's main window
class module_chooser(qw.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        outer_layout = qw.QVBoxLayout()

        buttons = [qw.QCheckBox("Input"), qw.QCheckBox("Module 1"),
                   qw.QCheckBox("Output"),]

        button = qw.QPushButton("Ok!")
        for b in buttons:
            outer_layout.addWidget(b)
        outer_layout.addWidget(button)

        # Set the central widget of the Window.
        self.setLayout(outer_layout)

if __name__ == '__main__':

    app = QApplication(sys.argv)

    window = ChooserWindow()
    window.show()

    app.exec()
