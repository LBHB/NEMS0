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
        super(LayerArea, self).__init__(parent)
        self.setupUi(self)

        self.layer_area_name = name
        # TODO: get the plot widget to respect wheel event propagation
        self.plotWidget.setParent(self)

        self.plotWidget.sigChannelsChanged.connect(self.update_spinbox)
        self.spinBox.valueChanged.connect(self.on_spinbox_changed)

    def update_spinbox(self, channels):
        self.spinBox.setRange(0, channels - 1)
        self.label.setText(f'channels: {channels}')

    def on_spinbox_changed(self, value):
        self.plotWidget.update_index(value)
