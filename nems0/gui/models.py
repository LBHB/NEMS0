import logging
import copy

import numpy as np
import PyQt5.QtWidgets as qw

log = logging.getLogger(__name__)


class ArrayModel(qw.QTableWidget):
    def __init__(self, parent, array):
        super(qw.QTableWidget, self).__init__(parent)
        self.array = array
        self.original_array = copy.deepcopy(array)
        while self.array.ndim < 2:
            self.array = np.expand_dims(self.array, 0)
        self.shape = self.array.shape
        self.ndims = self.array.ndim

        self._populate_table()
        self.itemChanged.connect(self._update_array)

    def _populate_table(self):
        nrows = self.shape[0]
        ncols = self.shape[1]
        self.setRowCount(nrows)
        self.setColumnCount(ncols)
        # Not the most efficient, but the arrays are generally going to be
        # pretty small so not going to bother with something more sophisticated.
        for i in range(nrows):
            for j in range(ncols):
                v = self.array[i, j]
                self.setItem(i, j, qw.QTableWidgetItem(str(v)))

    def _update_array(self, item):
        row = item.row()
        col = item.column()
        try:
            data = float(item.text())
        except ValueError:
            data = None

        if data is None:
            try:
                data = complex(item.text())
            except ValueError:
                data = None
        if data is None:
            data=0

        self.set(data, row, col)

    def _check_valid_coordinates(self, coords):
        for i, c in enumerate(coords):
            if (i > self.ndims-1) or (c > self.shape[i]):
                log.warning('coordinate #%s exceeded array dimensions', i)
                return False

        return True

    def set(self, value, *coords):
        if not self._check_valid_coordinates(coords):
            pass
        elif not np.isscalar(value) and np.isscalar(self.array[coords]):
            if not value.shape == self.array[coords].shape:
                log.warning("Can't assign value, shapes don't match for "
                             "%s and %s", value, self.array[coords])
        else:
            self.array[tuple(coords)] = value

    def get(self, *coords):
        if self._check_valid_coordinates(coords):
            return self.array[coords]

    def print(self):
        print(str(self.array))

    def export_array(self):
        if self.array.shape != self.original_array.shape:
            return np.full_like(self.original_array, self.array)
        else:
            return self.array
