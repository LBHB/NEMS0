import logging
import sys
from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from nems0.gui_new.model_browser.ui_promoted import PyPlotWidget, PG_PLOTS


log = logging.getLogger(__name__)

# read and parse the UI file to generate the GUI
qt_creator_file = Path(r'ui') / 'layer_area.ui'
#qt_creator_file = 'layer_area.ui'
Ui_Widget, QtBaseClass = uic.loadUiType(qt_creator_file)


class LayerArea(QtBaseClass, Ui_Widget):

    def __init__(self, rec_name, signal_names, parent=None):
        super(LayerArea, self).__init__(parent)
        self.setupUi(self)

        self.rec_name = rec_name
        self.signal_names = signal_names
        self.channels = 0
        self.plotWidget.setParent(self)

        self.plotWidget.sigChannelsChanged.connect(self.update_spinbox)
        self.spinBox.valueChanged.connect(self.on_spinbox_changed)
        self.comboBox.currentTextChanged.connect(self.on_combobox_changed)

        self.plot_type = 'pyqtgraph'

    def update_spinbox(self, channels):
        self.spinBox.setRange(0, channels - 1)
        self.label.setText(f'channels: {channels}')
        # self.spinBox.setValue(0)

    def on_spinbox_changed(self, value):
        self.channels = value
        self.plotWidget.update_index(value)

    def on_combobox_changed(self, text):
        if 'pyqtgraph' in text:
            new_plot = self.new_pg_plot(text)
        else:
            new_plot = self.new_nems_plot(text)

        self.window().unlink(self.plotWidget)
        self.layout().replaceWidget(self.plotWidget, new_plot)
        self.plotWidget.hide()
        del self.plotWidget

        self.plotWidget = new_plot
        self.plotWidget.sigChannelsChanged.connect(self.update_spinbox)
        self.update_plot(emit=True)
        # if updating can't keep up for pyplots, uncomment below, or add in a rate limiter on the signal
        self.window().link_together(self.plotWidget)  #, finished=self.plot_type == 'nems')

    def update_plot(self, rec_name=None, signal_names=None, channels=None, time_range=None, **kwargs):
        """Dispatches the proper plot calls."""
        if rec_name is not None:
            self.rec_name = rec_name
        if signal_names is not None:
            self.signal_names = signal_names
        if channels is not None:
            self.channels = channels
        if time_range is not None:
            self.time_range = time_range

        self.plotWidget.update_plot(rec_name=self.rec_name,
                                    signal_names=self.signal_names,
                                    channels=self.channels,
                                    **kwargs)

    def new_nems_plot(self, plot_fn):
        """Replaces the plot widget with a nems plot."""
        pyplot = PyPlotWidget(self, fn_path=plot_fn, modelspec=self.window().ctx['modelspec'])
        self.plot_type = 'nems'
        return pyplot

    def new_pg_plot(self, plot_fn):
        """Replaces the plot widget with a pyqtgraph plot."""
        pgplot = PG_PLOTS[plot_fn](self)
        self.plot_type = 'pyqtgraph'
        return pgplot
