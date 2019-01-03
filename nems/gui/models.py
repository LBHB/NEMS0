import logging
import copy

import numpy as np
import PyQt5.QtWidgets as qw

log = logging.getLogger(__name__)


class XformsModel(qw.QWidget):
    def __init__(self, parent, xfspec):
        raise NotImplementedError

        super(qw.QWidget, self).__init__(parent)
        self.xfspec = xfspec
        self.original_xfspec = copy.deepcopy(xfspec)
        self.shape = tuple([len(xf) for xf in self.xfspec])

    def set(self, value, *coords):
        if self._check_valid_coordinates(coords):
            self.xfspec[coords[0]][coords[1]] = value

    def get(self, *coords):
        if self._check_valid_coordinates(coords):
            return self.xfspec[coords[0]][coords[1]]
        else:
            return None

    def _check_valid_coordinates(self, coords):
        if len(coords) > 2:
            log.warning("Xfspec coordinates can't be greater than length 2")
            return False
        elif coords[0] > len(self.xfspec)-1:
            log.warning("First coordinate exceeded length of xfspec: %d/%d",
                        coords[0], len(self.xfspec)-1)
            return False
        elif coords[1] > len(self.xfspec[coords[0]])-1:
            log.warning("Second coordinate exceeded length of xf step: %d/%d",
                        coords[1], len(self.xfspec[coords[0]])-1)
            return False
        else:
            return True


class MspecModel(qw.QWidget):
    def __init__(self, parent, modelspec):
        raise NotImplementedError

        super(qw.QWidget, self).__init__(parent)
        self.modelspec = modelspec
        self.original_modelspec = copy.deepcopy(modelspec)
        self.shape = tuple([len(m['phi'].keys()) for m in self.modelspec])

    def set(self, value, *coords):
        if self._check_valid_coordinates(coords):
            key = list(self.modelspec[coords[0]].keys())[coords[1]]
            self.modelspec[coords[0]][key] = value

    def get(self, *coords):
        if self._check_valid_coordinates(coords):
            key = list(self.modelspec[coords[0]].keys())[coords[1]]
            return self.modelspec[coords[0]][key]
        else:
            return None

    def _check_valid_coordinates(self, coords):
        if len(coords) > 2:
            log.warning("Modelspec coordinates can't be greater than length 2")
            return False
        elif coords[0] > len(self.modelspec)-1:
            log.warning("First coordinate exceeded length of modelspec: %d/%d",
                        coords[0], len(self.modelspec)-1)
            return False
        elif coords[1] > len(self.modelspec[coords[0]].keys())-1:
            log.warning("Second coordinate exceeded length of module phi: %d/%d",
                        coords[1], len(self.modelspec[coords[0]])-1)
            return False
        else:
            return True


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
        data = float(item.text())
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


