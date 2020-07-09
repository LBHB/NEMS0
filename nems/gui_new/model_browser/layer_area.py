import logging
import sys
from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


log = logging.getLogger(__name__)

# read and parse the UI file to generate the GUI
qt_creator_file = Path(r'ui') / 'layer_area.ui'
Ui_Widget, QtBaseClass = uic.loadUiType(qt_creator_file)


class LayerArea(QtBaseClass, Ui_Widget):

    def __init__(self, parent=None, name=None):
        super(LayerArea, self).__init__()
        self.setupUi(self)

        self.layer_area_name = name
