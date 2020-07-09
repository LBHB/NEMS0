import logging
import sys
from pathlib import Path
from configparser import ConfigParser, DuplicateSectionError

from PyQt5 import uic
from PyQt5.QtWidgets import *

from nems.gui_new.model_browser.ui_promoted import CollapsibleBox
from nems.gui_new.model_browser.layer_area import LayerArea

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
qt_creator_file = Path(r'ui') / 'model_browser.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(qt_creator_file)


class MainWindow(QtBaseClass, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # add layout and collapsible boxes
        content = QWidget()
        self.scrollArea.setWidget(content)
        self.scrollArea.setWidgetResizable(True)

        vlayout = QVBoxLayout(content)

        for i in range(10):
            cbox = CollapsibleBox(f'Box #{i}')
            vlayout.addWidget(cbox)

            box_layout = QVBoxLayout()
            label = LayerArea(parent=content, name='testing')
            box_layout.addWidget(label)

            cbox.setContentLayout(box_layout)

        vlayout.addStretch()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
