from pathlib import Path
import logging
import sys
from configparser import ConfigParser, DuplicateSectionError

from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from nems import xforms, get_setting
from nems.gui_new.ui_promoted import ListViewModel

log = logging.getLogger(__name__)

# read and parse the UI file to generate the GUI
qt_creator_file = Path(r'ui\tab_comp.ui')
Ui_Widget, QtBaseClass = uic.loadUiType(qt_creator_file)

class CompTab(QtBaseClass, Ui_Widget):

    def __init__(self, parent=None):
        super(CompTab, self).__init__(parent)
        self.setupUi(self)
        self.parent = parent


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CompTab()
    window .show()
    app.exec_()