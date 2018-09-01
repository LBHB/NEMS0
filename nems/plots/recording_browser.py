#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:25:00 2018

@author: svd
"""

#!/usr/bin/env python

# embedding_in_qt5.py --- Simple Qt4 application embedding matplotlib canvases
#
# Copyright (C) 2005 Florent Rougon
#               2006 Darren Dale
# with Updates from @boxcontrol
# <http://www.boxcontrol.net/embedding-matplotlib-plot-on-pyqt5-gui.html>
# <https://github.com/boxcontrol/matplotlibPyQt5>
#
# This file is an example program for matplotlib. It may be used and
# modified with no restriction; raw copies as well as modified versions
# may be distributed without limitation.

import sys
import random
import copy

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QGridLayout, QPushButton, QScrollBar
from numpy import arange, sin, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import nems_db.db as nd
import nems_db.xform_wrappers as nw
import nems.xforms as xforms
import nems.plots.api as nplt
from nems.recording import Recording


class RecordingPlotWrapper():

    def __init__(self, recording):

        test_signals=['stim','resp','pred']

        self.recording = recording
        self.plot_signals = test_signals
        self.start_time = 0
        self.stop_time = 10

    def get_segment(self, signal):

        fs = self.recording[signal].fs
        start_bin = int(self.start_time * fs)
        stop_bin = int(self.stop_time * fs)

        d = recording[signal].as_continuous()[:,start_bin:stopbin]

        return d


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        self.compute_initial_figure()

        #
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class MyStaticMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""
    def compute_initial_figure(self):
        t = arange(0.0, 3.0, 0.01)
        s = sin(2*pi*t)
        self.axes.plot(t, s)


class MyDynamicMplCanvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""
    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(1000)

    def compute_initial_figure(self):
        self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')

    def update_figure(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        l = [random.randint(0, 10) for i in range(4)]

        self.axes.plot([0, 1, 2, 3], l, 'r')
        self.draw()


class nems_canvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""

    recording = None
    signal = []
    start_time = 0
    stop_time = 10

    def __init__(self, recording=None, signal='stim', *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        self.recording = recording
        self.signal = signal

    def compute_initial_figure(self):
        pass

    def update_figure(self):
        print('updating signal ' + self.signal)
        print('{} - {} '.format(self.start_time, self.stop_time))
        fs = self.recording[self.signal].fs
        start_bin = int(self.start_time * fs)
        stop_bin = int(self.stop_time * fs)

        d = self.recording[self.signal].as_continuous()[:,start_bin:stop_bin]

        t = np.linspace(self.start_time, self.stop_time, d.shape[1])
        self.axes.plot(t, d.T)
        self.draw()


class ApplicationWindow(QMainWindow):

    recording=None
    signals = []
    plot_list = []
    start_time = 0
    time_to_display = 10
    stop_time = 10
    time_slider = None

    def __init__(self, recording, signals=['stim','resp']):
        QMainWindow.__init__(self)

        self.recording=recording
        self.signals=signals

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.file_menu = QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

        self.main_widget = QWidget(self)

        # mpl panels
        layout = QGridLayout()
        self.plot_list = []
        for i in range(len(signals)):
            self.plot_list.append(nems_canvas(recording,
                                 signals[i], self.main_widget,
                                 width=5, height=4, dpi=100))
            layout.addWidget(self.plot_list[i], i, 0, 1, 2)

        # control buttons
        qbtn = QPushButton('Quit', self)
        qbtn.clicked.connect(self.close)

        qbtn2 = QPushButton('Test', self)
        qbtn2.clicked.connect(self.test)

        self.time_slider = QScrollBar(orientation = 1)
        self.time_slider.setMaximum(100)
        self.time_slider.valueChanged.connect(self.test)

        layout.addWidget(self.time_slider, len(signals), 0, 1, 2);

        layout.addWidget(qbtn, len(signals)+1, 0);
        layout.addWidget(qbtn2, len(signals)+1, 1);

        self.main_widget.setLayout(layout);

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("All hail matplotlib and NEMS!", 2000)

    def test(self):
        self.start_time = self.time_slider.value()
        self.stop_time = self.start_time + self.time_to_display

        for p in self.plot_list:
            p.start_time=self.start_time
            p.stop_time=self.stop_time
            p.update_figure()

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QMessageBox.about(self, "About",
  """embedding_in_qt5.py example
  Copyright 2015 BoxControL

  This program is a simple example of a Qt5 application embedding matplotlib
  canvases. It is base on example from matplolib documentation, and initially was
  developed from Florent Rougon and Darren Dale.

  http://matplotlib.org/examples/user_interfaces/embedding_in_qt4.html

  It may be used and modified with no restriction; raw copies as well as
  modified versions may be distributed without limitation."""
  )


batch=289
modelname="ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-fir.2x15-lvl.1-dexp.1_init-basic"
cellid='TAR010c-21-4'
#xf, ctx = nw.load_model_baphy_xform(cellid, batch, modelname,
#                                              eval_model=True)

#if __name__ == '__main__':
#    app = QApplication(sys.argv)
#
#    aw = ApplicationWindow()
#    aw.setWindowTitle("PyQt5 Matplot Example")
#    aw.show()
#    #sys.exit(qApp.exec_())
#    app.exec_()

aw = ApplicationWindow(recording=ctx['val'][0], signals=['stim','resp', 'pred'])
aw.setWindowTitle("NEMS data browser")
aw.show()
aw.test()

aw.raise_()