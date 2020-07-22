from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pyqtgraph as pg

import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm

pg.setConfigOptions(imageAxisOrder='row-major')
pg.setConfigOption('background', '#EAEAF2')
pg.setConfigOption('foreground', 'k')


class CollapsibleBox(QWidget):
    """Custom Collapsible Widget.

    Adapted from: https://stackoverflow.com/a/52617714
    """
    def __init__(self, parent=None):
        super(CollapsibleBox, self).__init__(parent)

        # self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        # self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        # self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        # self.toggle_button.setArrowType(Qt.RightArrow)
        # self.toggle_button.toggled.connect(self.on_pressed)

        self.toggle_animation = QParallelAnimationGroup(self)

        self.content_area = QScrollArea(parent=self, minimumHeight=0, maximumHeight=0)
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.content_area.setFrameShape(QFrame.NoFrame)

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        # layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_area)

        self.toggle_animation.addAnimation(QPropertyAnimation(self, b'minimumHeight'))
        self.toggle_animation.addAnimation(QPropertyAnimation(self, b'maximumHeight'))
        self.toggle_animation.addAnimation(QPropertyAnimation(self.content_area, b'maximumHeight'))

        self.is_open = False
        self.setMinimumHeight(0)
        self.setMaximumHeight(0)

    def set_toggle(self, toggle):
        """Toggles the cbox as needed."""
        if toggle != self.is_open:
            self.on_pressed()

    def on_pressed(self):
        # checked = self.toggle_button.isChecked()
        # self.toggle_button.setArrowType(Qt.DownArrow if not checked else Qt.RightArrow)
        self.toggle_animation.setDirection(
            QAbstractAnimation.Forward if not self.is_open
            else QAbstractAnimation.Backward
        )
        self.toggle_animation.start()
        self.is_open = False if self.is_open else True

    def setContentLayout(self, layout):
        self.content_area.setLayout(layout)
        collapsed_height = (self.sizeHint().height() - self.content_area.maximumHeight())
        content_height = layout.sizeHint().height()

        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(100)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        # content_animation = self.toggle_animation.animationAt(self.toggle_animation.animationCount() - 1)
        # content_animation.setDuration(250)
        # content_animation.setStartValue(0)
        # content_animation.setEndValue(content_height)


class PyPlotWidget(QWidget):

    def __init__(self, parent):
        super(PyPlotWidget, self).__init__(parent)
        layout = QVBoxLayout(self)

        self.canvas = FigureCanvas(Figure(figsize=(1, 1)))
        layout.addWidget(self.canvas)

        self.ax = self.canvas.figure.subplots()
        t = np.linspace(0, 10, 501)
        self.ax.plot(t, np.sin(t), '.')
        self.ax.figure.canvas.draw()

        self.ax.figure.set_size_inches = self.monkey_patch_figure_set_size_inches

    def monkey_patch_figure_set_size_inches(self, w, h=None, forward=True):
        """Monkey patches Figure.set_size_inches to accept heights <= 0."""
        self = self.ax.figure

        if h is None:  # Got called with a single pair as argument.
            w, h = w
        size = np.array([w, h])
        if not np.isfinite(size).all():
            raise ValueError(f'figure size must be positive finite not {size}')
        self.bbox_inches.p1 = size
        if forward:
            canvas = getattr(self, 'canvas')
            if canvas is not None:
                dpi_ratio = getattr(canvas, '_dpi_ratio', 1)
                manager = getattr(canvas, 'manager', None)
                if manager is not None:
                    manager.resize(*(size * self.dpi / dpi_ratio).astype(int))
        self.stale = True


def cmap_as_lut(name='viridis'):
    """Takes in a mpl cmap and returns a pyqtpgrah lut table."""
    colormap = cm.get_cmap(name)
    colormap._init()
    lut = (colormap._lut * 255).view(np.ndarray)
    return lut


class InputSpectrogram(pg.GraphicsLayoutWidget):

    def __init__(self,
                 *args,
                 **kwargs):
        super(InputSpectrogram, self).__init__(*args, **kwargs)

        viridis_lut = cmap_as_lut('viridis')

        self.p1 = self.addPlot()
        self.image_item_whole = pg.ImageItem()
        self.image_item_whole.setLookupTable(viridis_lut)

        self.lr = pg.LinearRegionItem([0, 500])
        # self.lr.setZValue(-10)

        self.p1.addItem(self.image_item_whole)
        self.p1.addItem(self.lr)
        self.p1.showAxis('left', False)
        self.p1.setMaximumHeight(50)

        self.nextRow()

        self.p2 = self.addPlot()
        self.image_item_sub = pg.ImageItem()
        self.image_item_sub.setLookupTable(viridis_lut)
        self.p2.addItem(self.image_item_sub, )
        self.p2.showAxis('left', False)
        self.ci.setSpacing(0)

        # self.updatePlot()
        self.p2.sigXRangeChanged.connect(self.updateRegion)
        self.lr.sigRegionChanged.connect(self.updatePlot)
        # TODO: add click event to move region instead of only dragging
        # i.e. that way if region goes off screen, can still move

    def plot_input(self, rec, mask=True, sig_name='stim'):
        """Plots a stim, applying a mask."""
        if mask and 'mask' in rec.signals:
            signal = rec.apply_mask()[sig_name]
        else:
            signal = rec[sig_name]

        self.image_item_whole.setImage(signal.as_continuous())
        self.image_item_sub.setImage(signal.as_continuous())

        self.p1.setLimits(xMin=0, xMax=signal.shape[1] - 1,
                          yMin=0, yMax=signal.shape[0] - 1,
                          minYRange=signal.shape[0] - 1)

        self.p2.setLimits(xMin=0, xMax=signal.shape[1] - 1,
                          yMin=0, yMax=signal.shape[0] - 1,
                          minYRange=signal.shape[0] - 1)
        # self.p2.setLabel('bottom', 'test', units='s')

        self.p1.setXRange(0, signal.shape[1] - 1)
        self.p1.setYRange(0, signal.shape[0] - 1)
        self.p2.setYRange(0, signal.shape[0] - 1)
        self.lr.setRegion([0, 500])

    def updatePlot(self, sender):
        """When the region item is changed, update the lower plot to match."""
        self.p2.setXRange(*sender.getRegion(), padding=0)

    def updateRegion(self, sender):
        """When the plot item XRange is changed, update the upper region to match."""
        self.lr.setRegion(sender.viewRange()[0])

    def add_link(self, fn):
        self.lr.sigRegionChanged.connect(fn)


class OutputPlot(pg.PlotWidget):

    # seaborn color palette as hex
    # TODO: add alpha?
    BLUE = '#4c72b0'
    GREEN = '#55a868'
    RED = '#c44e52'
    ORANGE = '#dd8452'
    BRIGHT_ORANGE = '#ff7c00'

    # signal for parent to hook into
    sigChannelsChanged = pyqtSignal(int)

    def __init__(self,
                 parent,
                 y_data=None,
                 y_data2=None,
                 y_idx=0,
                 x_link=None,
                 *args,
                 **kwargs):
        """Both y_datas much be same shape.

        y1 is pred, y2 is resp.
        """
        super(OutputPlot, self).__init__(parent, *args, **kwargs)

        self.legend = self.addLegend(offset=None, verSpacing=-10)
        self.legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(5, -10))

        self.line = self.plot(pen=self.ORANGE, name='pred')
        self.line2 = self.plot(pen=self.BLUE, name='resp')

        self.update_plot(y_data, y_data2, y_idx=y_idx)

        if x_link is not None:
            self.set_xlink(x_link)

    def set_xlink(self, x_link):
        self.plotItem.vb.setXLink(x_link)

    def add_link(self, fn):
        self.sigXRangeChanged.connect(fn)

    def update_index(self, y_idx=0):
        """Updates just the index."""
        self.update_plot(self.y_data, self.y_data2, y_idx, emit=False)

    def updateXRange(self, sender):
        """When the region item is changed, update the lower plot to match."""
        self.setXRange(*sender.getRegion(), padding=0)

    def update_plot(self, y_data, y_data2=None, y_idx=0, emit=True):
        """Updates members and plots."""
        self.y_data = y_data
        self.y_data2 = y_data2
        self.y_idx = y_idx

        y2 = None

        # messy nested ifs, but it works
        if self.y_data is not None:
            if self.y_data.ndim == 2:
                y = self.y_data[self.y_idx]
                if self.y_data2 is not None:
                    if self.y_data2.ndim != 2:
                        raise ValueError('y_data and y_data2 must have same dims')
                    y2 = self.y_data2[self.y_idx]
            else:
                y = self.y_data
                y2 = self.y_data2
        else:
            y = None

        self.line.setData(y=y)
        self.line2.setData(y=y2)

        if y is not None:
            self.setLimits(xMin=0, xMax=len(y))
            # self.setXRange(0, 500)

            # update the spinbox if possible
            if emit:
                self.sigChannelsChanged.emit(self.y_data.shape[0])


class DockTitleBar(QWidget):
    def __init__(self, dockWidget, title=''):
        super(DockTitleBar, self).__init__(dockWidget)

        boxLayout = QHBoxLayout(self)
        boxLayout.setSpacing(1)
        boxLayout.setContentsMargins(1, 1, 1, 1)

        self.titleLabel = QLabel(title)

        iconSize = QApplication.style().standardIcon(
            QStyle.SP_TitleBarNormalButton).actualSize(
                QSize(12, 12))
        buttonSize = iconSize + QSize(2, 2)

        self.minButton = QToolButton(self)
        self.minButton.setMaximumSize(buttonSize)
        self.minButton.setAutoRaise(True)
        self.minButton.setIcon(QApplication.style().standardIcon(
            QStyle.SP_TitleBarMinButton))
        # self.minButton.clicked.connect(self.toggleFloating)

        self.dockButton = QToolButton(self)
        self.dockButton.setMaximumSize(buttonSize)
        self.dockButton.setAutoRaise(True)
        self.dockButton.setIcon(QApplication.style().standardIcon(
            QStyle.SP_TitleBarNormalButton))
        self.dockButton.clicked.connect(self.toggleFloating)

        self.closeButton = QToolButton(self)
        self.closeButton.setMaximumSize(buttonSize)
        self.closeButton.setAutoRaise(True)
        self.closeButton.setIcon(QApplication.style().standardIcon(
            QStyle.SP_TitleBarCloseButton))
        self.closeButton.clicked.connect(self.closeParent)

        boxLayout.addSpacing(5)
        boxLayout.addWidget(self.titleLabel)
        boxLayout.addStretch()
        boxLayout.addSpacing(5)
        boxLayout.addWidget(self.minButton)
        boxLayout.addWidget(self.dockButton)
        boxLayout.addWidget(self.closeButton)

        dockWidget.featuresChanged.connect(self.onFeaturesChanged)

        self.onFeaturesChanged(dockWidget.features())

    def onFeaturesChanged(self, features):
        if not features & QDockWidget.DockWidgetVerticalTitleBar:
            self.closeButton.setVisible(
                features & QDockWidget.DockWidgetClosable)
            self.dockButton.setVisible(
                features & QDockWidget.DockWidgetFloatable)
        else:
            raise ValueError('vertical title bar not supported')

    def toggleFloating(self):
        self.parent().setFloating(not self.parent().isFloating())

    def closeParent(self, ev):
        self.parent().close()

    def mouseReleaseEvent(self, event):
        event.ignore()

    def mousePressEvent(self, event):
        event.ignore()

    def mouseMoveEvent(self, event):
        event.ignore()


class LeftDockWidget(QDockWidget):

    def __init__(self, *args, window_title='', dock_title=None, **kwargs):
        super(LeftDockWidget, self).__init__(*args, **kwargs)

        if dock_title is None:
            dock_title = window_title

        self.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.setFloating(False)

        self.layout().setContentsMargins(0, 0, 0, 0)

        self.setTitleBarWidget(DockTitleBar(self, title=dock_title))
        self.setWindowTitle(window_title)

    def connect_min(self, fn):
        """Connects the min button to a custom callback."""
        self.titleBarWidget().minButton.clicked.connect(fn)
