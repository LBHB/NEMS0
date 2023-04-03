from typing import List, Union
import itertools as it

import numpy as np
import scipy.spatial

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pyqtgraph as pg

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib import pyplot as plt

from sqlalchemy.orm import aliased

from nems0 import db
from nems0.utils import lookup_fn_at

pg.setConfigOptions(imageAxisOrder='row-major')
pg.setConfigOption('background', '#EAEAF2')
pg.setConfigOption('foreground', 'k')


class ListViewListModel(QAbstractListModel):

    def __init__(self,
                 table_name: str,
                 column_name: str,
                 filter=None,
                 *args,
                 **kwargs
                 ):
        super(ListViewListModel, self).__init__(*args, **kwargs)

        self.table_name = table_name
        self.column_name = column_name

        self.data = None

        self.init_db()
        self.refresh_data(filters=filter)

    def data(self, index, role):
        if role == Qt.DisplayRole or role == Qt.EditRole:
            text = self.data[index.row()][0]
            return text

    def rowCount(self, index):
        return len(self.data)

    def init_db(self):
        self.engine = db.Engine()
        table = db.Tables()[self.table_name]
        self.column = getattr(table, self.column_name)
        self.session = db.sessionmaker(bind=self.engine)()

    def close_db(self):
        self.session.close()
        self.engine.dispose()

    def refresh_data(self, filters: dict = None):
        """Reads the data from the database, optionally filtering.

        # TODO: if app is idle too long, db connection drops. Need a try catch for this and reinit the db on loss.

        :param filters: Dict of column: filter.
        :return:
        """
        query = (self.session
                 .query(self.column)
                 .distinct()
                 )

        if filters is not None:
            query = query.filter_by(**filters)

        query = query.order_by(self.column)
        data = query.all()

        self.data = data

    def get_index(self, value) -> int:
        """Searches for a value in the data, returns the index"""
        flat = [i[0] for i in self.data]
        try:
            return flat.index(value)
        except ValueError:
            return -1


class ListViewTableModel(QAbstractTableModel):

    def __init__(self,
                 table_name: str,
                 column_names: List[str],
                 filter=None,
                 *args,
                 **kwargs
                 ):
        super(ListViewListModel, self).__init__(*args, **kwargs)

        self.table_name = table_name
        self.column_names = column_names

        self.data = None

        self.init_db()
        self.refresh_data(filters=filter)

    def data(self, index, role):
        if role == Qt.DisplayRole or role == Qt.EditRole:
            text = self.data[index.row()][index.column()]
            return text

    def rowCount(self, index):
        return len(self.data)

    def columnCount(self, index):
        return len(self.data[0])

    def init_db(self):
        self.engine = db.Engine()
        table = db.Tables()[self.table_name]
        self.columns = [getattr(table, column_name) for column_name in self.column_names]
        self.session = db.sessionmaker(bind=self.engine)()

    def close_db(self):
        self.session.close()
        self.engine.dispose()

    def refresh_data(self, filters: dict = None, order_by: Union[List[str], None] = None):
        """Reads the data from the database, optionally filtering.

        :param filters: Dict of {column: filter}.
        :param order_by: List of columns to order by.
        """
        query = (self.session
                 .query(*self.columns)
                 # .distinct()
                 )

        if filters is not None:
            query = query.filter_by(**filters)

        if order_by is None:
            order_by = []

        if order_by:
            for ob in order_by:
                query = query.order_by(ob)
        data = query.all()

        self.data = data

    def get_index(self, value) -> int:
        """Searches for a value in the data, returns the index"""
        flat = [i[0] for i in self.data]
        try:
            return flat.index(value)
        except ValueError:
            return -1


class CompModel(QAbstractTableModel):
    """Reimplements refresh_data() for some custom joins to do the equivalent of pd.unstack()."""
    def __init__(self,
                 filter1,
                 filter2,
                 table_name: str,
                 *args,
                 merge_on='cellid',
                 comp_val='r_test',
                 filter_on='modelname',
                 filters=None,
                 **kwargs):
        super(CompModel, self).__init__(*args, **kwargs)

        self.table_name = table_name
        self.init_db()

        # need to alias the table
        self.alias_a = aliased(self.table)
        self.alias_b = aliased(self.table)

        self.merge_on = merge_on
        self.comp_val = comp_val
        self.filter_on = filter_on

        self.data = None
        self.np_points = None
        self.np_labels = None

        self.refresh_data(filter1, filter2, filters)

    def data(self, index, role):
        if role == Qt.DisplayRole:
            text = self.data[index.row()][index.column()]
            return text

    def rowCount(self, index):
        return len(self.data)

    def columnCount(self, index):
        if len(self.data) == 0:
            return 0
        return len(self.data[0])

    def init_db(self):
        self.engine = db.Engine()
        self.table = db.Tables()[self.table_name]
        self.session = db.sessionmaker(bind=self.engine)()

    def close_db(self):
        self.session.close()
        self.engine.dispose()

    def refresh_data(self, filter1, filter2, filters=None):
        merge_on_a = getattr(self.alias_a, self.merge_on)
        merge_on_b = getattr(self.alias_b, self.merge_on)

        comp_a = getattr(self.alias_a, self.comp_val)
        comp_b = getattr(self.alias_b, self.comp_val)

        filter_a = getattr(self.alias_a, self.filter_on)
        filter_b = getattr(self.alias_b, self.filter_on)

        query = (self.session
                 .query(
                     merge_on_a.label(self.merge_on),
                     comp_a.label(self.comp_val + '_one'),
                     comp_b.label(self.comp_val + '_two'))
                 .filter(
                     filter_a == filter1,
                     filter_b == filter2,
                     filter_a != filter_b,
                     merge_on_a == merge_on_b)
                 )

        for key, value in filters.items():
            fixed_a = getattr(self.alias_a, key)
            fixed_b = getattr(self.alias_b, key)
            query = query.filter(
                fixed_a == value,
                fixed_b == value,
            )

        query = query.order_by(merge_on_a)

        data = query.all()
        self.data = np.asarray(data)

        if self.data.size:
            self.np_points = np.asarray(self.data).T
            self.np_labels = self.np_points[0]
            self.np_points = self.np_points[1:].astype('float')
        else:
            self.np_labels = None
            self.np_points = None


class CompPlotWidget(pg.PlotWidget):

    # seaborn color palette as hex
    BLUE = '#4c72b0'
    GREEN = '#55a868'
    RED = '#c44e52'
    ORANGE = '#dd8452'
    BRIGHT_ORANGE = '#ff7c00'

    def __init__(self,
                 left_label=None,
                 bottom_label=None,
                 x_data=None,
                 y_data=None,
                 labels=None,
                 units=None,
                 *args,
                 **kwargs):
        super(CompPlotWidget, self).__init__(*args, **kwargs)

        self.setAspectLocked(True)

        self.left_label = None
        self.bottom_label = None
        self.units = None
        self.x_data = None
        self.y_data = None
        self.labels = None
        self.colors = None
        self.ckdtree = None
        self.nearest_idx = None

        unity = pg.InfiniteLine((0, 0), angle=45, pen='k')
        self.addItem(unity)

        self.scatter = self.plot(pen=None, symbolSize=5, symbolPen='w')
        self.getPlotItem().getAxis('left').enableAutoSIPrefix(False)
        self.getPlotItem().getAxis('bottom').enableAutoSIPrefix(False)

        self.showGrid(x=True, y=True, alpha=0.2)

        if left_label is not None and bottom_label is not None:
            self.set_labels(left_label, bottom_label, units)

        if x_data is not None and y_data is not None:
            self.update_data(x_data, y_data, labels)

        # upper left text to display nearest point
        self.label_text = pg.LabelItem(text='', justify='left')
        self.label_text.setParentItem(self.plotItem.vb)
        self.label_text.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(10, 5))

        # bottom right text to display mouse coords
        self.coord_text = pg.LabelItem(text='', justify='right')
        self.coord_text.setParentItem(self.plotItem.vb)
        self.coord_text.anchor(itemPos=(1, 1), parentPos=(1, 1), offset=(-10, -5))

        # connections for the hover data
        self.plotItem.scene().sigMouseMoved.connect(self.on_move)

    def set_labels(self, label_left, label_bottom, units=None):
        """Helper to simplify setting labels."""
        self.left_label = label_left
        self.bottom_label = label_bottom

        if self.units is None and units is not None:
            self.units = units
        self.setLabel('left', label_left, units=self.units)
        self.setLabel('bottom', label_bottom, units=self.units)
        # need to trigger resize events to get label to recenter
        self.getPlotItem().getAxis('left').resizeEvent()
        self.getPlotItem().getAxis('bottom').resizeEvent()

    def clear_data(self):
        """Sets the scatter data to nothing."""
        self.x_data = []
        self.y_data = []
        self.ckdtree = None

        self.scatter.setData(x=[], y=[], symbolBrush=[])

    def update_data(self, x, y, labels=None, size=6):
        """Handles the logic for plotting, such as colors."""
        self.x_data = x
        self.y_data = y
        self.labels = labels

        self.scatter.setData(x=x, y=y, symbolSize=size, symbolBrush='k')
        self.update_colors()
        self.build_ckdt()

    def update_colors(self, selected: int = None):
        """Updates the colors. If values are same, defaults to green.

        TODO: default to other color? Brings total colors to 4...
        """
        colors = np.full_like(self.x_data, self.BLUE, dtype='object')
        colors[self.x_data > self.y_data] = self.RED
        colors[self.y_data > self.x_data] = self.GREEN
        if selected is not None:
            colors[selected] = self.BRIGHT_ORANGE

        colors = [pg.mkColor(c) for c in colors]
        self.colors = colors
        self.scatter.setData(symbolBrush=colors)

    def get_copy(self):
        widget = CompPlotWidget(
            left_label=self.left_label,
            bottom_label=self.bottom_label,
            x_data=self.x_data,
            y_data=self.y_data,
            labels=self.labels,
            units=self.units,
        )

        return widget

    def on_move(self, pos):
        """Handles mouse movement on plot events."""
        if self.plotItem.vb.sceneBoundingRect().contains(pos):
            mapped_pos = self.plotItem.vb.mapSceneToView(pos)
            x, y = mapped_pos.x(), mapped_pos.y()
            self.coord_text.setText(f'x: {x: 04.3f} | y: {y: 04.3f}')
            # get nearest scatter point
            if self.ckdtree is not None:
                nearest_idx = self.ckdtree.query([x, y])[1]
                if self.nearest_idx != nearest_idx:
                    self.update_nearest(nearest_idx)
        else:
            self.coord_text.setText('')
            self.label_text.setText('')
            self.update_colors()

    def build_ckdt(self):
        """Builds a nearest tree for the hover event.

        See scipy.spatial.cKDTree"""
        points = np.column_stack([self.x_data, self.y_data])
        self.ckdtree = scipy.spatial.cKDTree(points)

    def update_nearest(self, nearest_idx):
        """Highlights the point and updates the label"""
        self.nearest_idx = nearest_idx
        scatter_x, scatter_y = self.x_data[nearest_idx], self.y_data[nearest_idx]
        self.label_text.setText(f'{self.labels[nearest_idx]}<br>x: {scatter_x: 04.3f} | y: {scatter_y: 04.3f}')
        self.update_colors(selected=self.nearest_idx)



class ExtendedComboBox(QComboBox):

    def setModel(self, model):
        super(ExtendedComboBox, self).setModel(model)
        self.completer().setCompletionMode(QCompleter.PopupCompletion)
        self.completer().setFilterMode(Qt.MatchContains)
        self.completer().setCompletionRole(Qt.DisplayRole)



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
                          yMin=0, yMax=signal.shape[0],
                          minYRange=signal.shape[0] - 1)

        self.p2.setLimits(xMin=0, xMax=(signal.shape[1] - 1) / fs,
                          yMin=0, yMax=signal.shape[0],
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
