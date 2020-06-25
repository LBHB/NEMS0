from typing import List, Union

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import pyqtgraph as pg

from sqlalchemy.orm import aliased
import numpy as np

from nems import db


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
        if role == Qt.DisplayRole:
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
        if role == Qt.DisplayRole:
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
        self.np_data = None
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

        if filters is None:
            filters = {}

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
                 .filter_by(
                     **filters)
                 .order_by(
                     merge_on_a)
                 )

        data = query.all()
        self.data = data

        if data:
            self.np_data = np.asarray(data).T
            self.np_labels = self.np_data[0]
            self.np_data = self.np_data[1:].astype('float')
        else:
            self.np_labels = None
            self.np_data = None


class CompPlotWidget(pg.PlotWidget):

    # seaborn color palette as hex
    BLUE = '#4c72b0'
    GREEN = '#55a868'
    RED = '#c44e52'

    def __init__(self, *args, **kwargs):
        super(CompPlotWidget, self).__init__(*args, **kwargs)

        self.setAspectLocked(True)
        self.units = None

        unity = pg.InfiniteLine((0, 0), angle=45, pen='k')
        self.addItem(unity)

        self.scatter = self.plot(pen=None, symbolSize=5, symbolPen='w')
        self.getPlotItem().getAxis('left').enableAutoSIPrefix(False)
        self.getPlotItem().getAxis('bottom').enableAutoSIPrefix(False)

    def set_labels(self, label_left, label_bottom, units=None):
        """Helper to simplify setting labels."""
        if self.units is None and units is not None:
            self.units = units
        self.setLabel('left', label_left, units=self.units)
        self.setLabel('bottom', label_bottom, units=self.units)
        # need to trigger resize events to get label to recenter
        self.getPlotItem().getAxis('left').resizeEvent()
        self.getPlotItem().getAxis('bottom').resizeEvent()

    def clear_data(self):
        """Sets the scatter data to nothing."""
        self.scatter.setData(x=[], y=[], symbolBrush=[])

    def update_data(self, x, y, labels=None, size=6):
        """Handles the logic for plotting, such as colors."""
        colors = np.full_like(x, self.BLUE, dtype='object')
        colors[x > y] = self.RED
        colors[y > x] = self.GREEN
        colors = [pg.mkColor(c) for c in colors]
        self.scatter.setData(x=x, y=y, symbolSize=size, symbolBrush=colors)


class ExtendedComboBox(QComboBox):

    def __init__(self, parent=None):
        super(ExtendedComboBox, self).__init__(parent)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setEditable(True)

        # add a filter model to filter matching items
        self.pFilterModel = QSortFilterProxyModel(self)
        self.pFilterModel.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.pFilterModel.setSourceModel(self.model())

        # add a completer, which uses the filter model
        self.completer = QCompleter(self.pFilterModel, self)
        # always show all (filtered) completions
        self.completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        self.setCompleter(self.completer)

        # connect signals
        self.lineEdit().textEdited.connect(self.pFilterModel.setFilterFixedString)
        self.completer.activated.connect(self.on_completer_activated)

    def on_completer_activated(self, text):
        if text:
            index = self.findText(text)
            self.setCurrentIndex(index)
            self.activated[str].emit(self.itemText(index))

    def setModel(self, model):
        super(ExtendedComboBox, self).setModel(model)
        self.pFilterModel.setSourceModel(model)
        self.completer.setModel(self.pFilterModel)

    def setModelColumn(self, column):
        self.completer.setCompletionColumn(column)
        self.pFilterModel.setFilterKeyColumn(column)
        super(ExtendedComboBox, self).setModelColumn(column)
