from pathlib import Path
import logging
import sys

from PyQt5 import uic
from PyQt5.QtWidgets import *

# TEMP ERROR CATCHER
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook


def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


# Set the exception hook to our wrapping function
sys.excepthook = my_exception_hook

log = logging.getLogger(__name__)

# read and parse the UI file to generate the GUI
qt_creator_file = Path(r'ui\container.ui')
Ui_MainWindow, QtBaseClass = uic.loadUiType(qt_creator_file)


class MainWindow(QtBaseClass, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # setup the callbacks for the menu items
        self.actionOpen.triggered.connect(self.tabBrowser.on_action_open)
        self.actionSave_selections.triggered.connect(self.tabBrowser.on_action_save_selections)
        self.actionLoad_selections.triggered.connect(self.tabBrowser.on_action_load_selections)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window .show()
    app.exec_()
