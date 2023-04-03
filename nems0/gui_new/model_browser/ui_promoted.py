import itertools as it

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pyqtgraph as pg

import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib import pyplot as plt

from nems0.utils import lookup_fn_at

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

    # signal for parent to hook into
    sigChannelsChanged = pyqtSignal(int)

    def __init__(self, parent, fn_path=None, modelspec=None):
        super(PyPlotWidget, self).__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fn_path = fn_path
        self.modelspec = modelspec
        self.rec_name = None
        self.signal_names = None
        self.channels = None
        self.time_range = None

        # need to name the figure so we can set it to the current figure before passing to nems plotting functions
        self.figure = plt.figure(f'{id(self)}', figsize=(1, 1))
        self.figure.clf()  # also need to clear it, since it might an existing figure
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.ax = self.canvas.figure.subplots()

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

    def update_index(self, value=0, **kwargs):
        """Updates just the index."""
        self.update_plot(
            channels=value,
            emit=False,
            **kwargs)

    def update_plot(self, fn_path=None, modelspec=None, rec_name=None, signal_names=None, channels=None, time_range=None,
                    emit=True,
                    **kwargs):
        """Updates members and plots."""
        if fn_path is not None:
            self.fn_path = fn_path
        if modelspec is not None:
            self.modelspec = modelspec
        if rec_name is not None:
            self.rec_name = rec_name
        if signal_names is not None:
            if len(signal_names) != 1:
                raise ValueError('NEMS can only plot a single signal.')
            self.signal_names = signal_names
        if channels is not None:
            self.channels = channels
        if time_range is not None:
            self.time_range = time_range

        plot_fn = lookup_fn_at(self.fn_path)
        plt.figure(f'{id(self)}')
        self.ax.clear()

        # fill the area
        pos = self.ax.get_position()
        pos.x0 = 0.1
        pos.x1 = 1
        self.ax.set_position(pos)

        rec = self.window().rec_container[self.rec_name]

        # sometimes the current channel index can be out of range
        try:
            plot_fn(rec=rec,
                    modelspec=self.modelspec,
                    sig_name=self.signal_names[0],
                    ax=self.ax,
                    channels=channels,
                    time_range=self.time_range,
                    **kwargs)
            self.canvas.figure.canvas.draw()
        except IndexError:
            self.parent().spinBox.setValue(0)
            return

        if emit:
            # TODO: how to know how many channels can be viewed here?
            #  but might not matter since plotting fn can handle channel input?
            self.sigChannelsChanged.emit(rec[self.signal_names[0]].shape[0])
            # has to be a better way to do this

    def add_link(self, fn):
        # self.sigXRangeChanged.connect(fn)
        pass

    def unlink(self, fn):
        # self.sigXRangeChanged.disconnect(fn)
        pass

    def updateXRange(self, sender):
        """When the region item is changed, update the lower plot to match."""
        # self.setXRange(*sender.getRegion(), padding=0)
        self.update_plot(time_range=sender.getRegion())
        # print(f'nems update: {sender.getRegion()}')
        # pass


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
        self.p1.scene().sigMouseClicked.connect(self.on_mouse_click)

        # self.proxies = []  # container to persist instances of SignalProxy

    def plot_input(self, rec_name, mask=True, sig_name='stim'):
        """Plots a stim, applying a mask."""
        rec = self.window().rec_container[rec_name]
        fs = rec[sig_name].fs

        if mask and 'mask' in rec.signals:
            signal = rec.apply_mask()[sig_name]
        else:
            signal = rec[sig_name]

        self.image_item_whole.setImage(signal.as_continuous())
        self.image_item_sub.setImage(signal.as_continuous())

        # scale in x, y, z
        self.image_item_whole.scale(1 / fs, 1)
        self.image_item_sub.scale(1 / fs, 1)

        self.p1.setLimits(xMin=0, xMax=(signal.shape[1] - 1) / fs,
                          yMin=0, yMax=signal.shape[0] - 1,
                          minYRange=signal.shape[0] - 1)

        self.p2.setLimits(xMin=0, xMax=(signal.shape[1] - 1) / fs,
                          yMin=0, yMax=signal.shape[0] - 1,
                          minYRange=signal.shape[0] - 1)
        # self.p2.setLabel('bottom', 'test', units='s')

        self.p1.setXRange(0, (signal.shape[1] - 1) / fs)
        self.p1.setYRange(0, signal.shape[0] - 1)
        self.p2.setYRange(0, signal.shape[0] - 1)
        self.lr.setRegion([0, 500 / fs])

    def updatePlot(self, sender):
        """When the region item is changed, update the lower plot to match."""
        self.p2.setXRange(*sender.getRegion(), padding=0)

    def updateRegion(self, sender):
        """When the plot item XRange is changed, update the upper region to match."""
        self.lr.setRegion(sender.viewRange()[0])

    def add_link(self, fn, finished=False):
        """Connects region changed event.

        :param finished: Whether to trigger during whole drag, or just when dragging finished."""
        if finished:
            self.lr.sigRegionChangeFinished.connect(fn)
        else:
            self.lr.sigRegionChanged.connect(fn)

        # self.proxies.append(pg.SignalProxy(self.lr.sigRegionChanged, delay=0, rateLimit=30, slot=fn))

    def unlink(self, fn):
        """Disconnects region change event."""
        try:
            self.lr.sigRegionChanged.disconnect(fn)
        except TypeError:
            pass
        try:
            self.lr.sigRegionChangeFinished.disconnect(fn)
        except TypeError:
            pass

    def on_mouse_click(self, ev):
        """Reposition the region on click."""
        if self.p1.vb in self.p1.scene().items(ev.scenePos()):
            # get the current width of the region, then center it around the click
            region = self.lr.getRegion()
            half_width = (region[1] - region[0]) / 2
            click_x_coord = self.p1.vb.mapSceneToView(ev.scenePos()).x()
            self.lr.setRegion([click_x_coord - half_width, click_x_coord + half_width])
            ev.accept()


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
                 rec_name=None,
                 signal_names=None,
                 channels=None,
                 time_range=None,
                 x_link=None,
                 *args,
                 **kwargs):
        """Both y_datas much be same shape.

        y1 is pred, y2 is resp.
        """
        super(OutputPlot, self).__init__(parent, *args, **kwargs)

        self.color_cycle = it.cycle([self.ORANGE,
                                     self.BLUE,
                                     self.GREEN,
                                     self.RED])

        self.legend = self.addLegend(offset=None, verSpacing=-10)
        self.legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(5, -10))

        self.lines = []

        self.rec_name = rec_name
        self.signal_names = signal_names
        self.channels = channels
        self.time_range = time_range

    def add_link(self, fn):
        self.sigXRangeChanged.connect(fn)

    def unlink(self, fn):
        try:
            self.sigXRangeChanged.disconnect(fn)
        except TypeError:
            pass

    def update_index(self, value=0, **kwargs):
        """Updates just the index."""
        self.update_plot(
            channels=value,
            emit=False,
            **kwargs)

    def updateXRange(self, sender):
        """When the region item is changed, update the lower plot to match."""
        self.setXRange(*sender.getRegion(), padding=0)

    def update_plot(self, rec_name=None, signal_names=None, channels=None, time_range=None, emit=True, **kwargs):
        """Updates members and plots."""
        if rec_name is not None:
            self.rec_name = rec_name
        if signal_names is not None:
            self.signal_names = signal_names
        if time_range is not None:
            self.time_range = time_range

        if not self.lines:
            self.lines = [
                self.plot(pen=color)  # zip this way to get color cycle matching len of signals
                for signal, color in zip(self.signal_names, self.color_cycle)
            ]

        assert len(self.lines) == len(self.signal_names)
        self.legend.clear()

        for line, signal_name in zip(self.lines, self.signal_names):
            signal = self.window().rec_container[self.rec_name][signal_name]

            y_data = signal.as_continuous()
            x_data = np.arange(0, y_data.shape[-1]) / signal.fs

            line.setData(y=y_data[channels], x=x_data)
            self.legend.addItem(line, signal_name)

        # bad practice, but use the last reference for the for loop
        self.setLimits(xMin=0, xMax=y_data.shape[-1] - 1)

        if emit:
            self.sigChannelsChanged.emit(y_data.shape[0])


class SpectrogramPlot(pg.PlotWidget):

    # signal for parent to hook into
    sigChannelsChanged = pyqtSignal(int)

    def __init__(self,
                 parent,
                 rec_name=None,
                 signal_names=None,
                 channels=None,
                 time_range=None,
                 x_link=None,
                 *args,
                 **kwargs):
        """Both y_datas much be same shape.

        y1 is pred, y2 is resp.
        """
        super(SpectrogramPlot, self).__init__(parent, *args, **kwargs)

        self.image_item = pg.ImageItem()
        self.addItem(self.image_item)

        viridis_lut = cmap_as_lut('viridis')
        self.image_item.setLookupTable(viridis_lut)

        self.rec_name = rec_name
        self.signal_names = signal_names
        self.channels = channels
        self.time_range = time_range

    def add_link(self, fn):
        self.sigXRangeChanged.connect(fn)

    def unlink(self, fn):
        self.sigXRangeChanged.disconnect(fn)

    def update_index(self, value=0):
        """Updates just the index."""
        pass

    def updateXRange(self, sender):
        """When the region item is changed, update the lower plot to match."""
        self.setXRange(*sender.getRegion(), padding=0)

    def update_plot(self, rec_name=None, signal_names=None, channels=None, time_range=None, emit=True, **kwargs):
        """Updates members and plots."""
        if rec_name is not None:
            self.rec_name = rec_name
        if signal_names is not None:
            if len(signal_names) != 1:
                raise ValueError('Spectrogram can only plot a single signal.')
            self.signal_names = signal_names
        if time_range is not None:
            self.time_range = time_range

        # TODO: do we need to apply mask, or is that done already?
        signal = self.window().rec_container[self.rec_name][self.signal_names[0]]
        fs = signal.fs

        image_data = signal.as_continuous()
        self.image_item.scale(1 / fs, 1)

        self.image_item.setImage(image_data)

        # bad practice, but use the last reference for the for loop
        self.setLimits(xMin=0, xMax=(image_data.shape[-1] - 1) / fs,
                       yMin=0, yMax=image_data.shape[0] - 1,
                       minYRange=image_data.shape[0] - 1)

        if emit:
            self.sigChannelsChanged.emit(1)


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


PG_PLOTS = {
    'pyqtgraph output plot': OutputPlot,
    'pyqtgraph spectrogram': SpectrogramPlot,
}
