import sys
import copy
import json
import logging

import numpy as np
import PyQt5.QtWidgets as qw
import PyQt5.QtCore as qc

from nems import xforms
from nems.gui.models import ArrayModel

log = logging.getLogger(__name__)


class ModelEditor(qw.QMainWindow):
    def __init__(self, ctx=None, xfspec=None, parent=None):
        qw.QMainWindow.__init__(self)
        self.ctx = ctx
        self.xfspec = xfspec
        self.original_ctx = copy.deepcopy(ctx)
        self.original_xfspec = copy.deepcopy(xfspec)
        self.parent = parent

        self.title = 'NEMS Model Editor'

        self.editor = EditorWidget(self)
        self.setCentralWidget(self.editor)

        self.show()

    def update_browser(self):
        # TODO: after updating xfspec / ctx, tell model browser to
        #       update plots
        pass


class EditorWidget(qw.QWidget):
    def __init__(self, parent):
        super(qw.QWidget, self).__init__(parent)
        self.parent = parent
        self.layout = qw.QVBoxLayout(self)
        self.tabs = qw.QTabWidget(self)
        self.xfspec_tab = qw.QWidget()
        self.modelspec_tab = qw.QWidget()
        self.tabs.addTab(self.xfspec_tab, 'xfspec')
        self.tabs.addTab(self.modelspec_tab, 'modelspec')

        self.xfspec_tab.layout = self._xfspec_setup()
        self.xfspec_tab.setLayout(self.xfspec_tab.layout)

        self.modelspec_tab.layout = self._modelspec_setup()
        self.modelspec_tab.setLayout(self.modelspec_tab.layout)

        self.reset = qw.QPushButton('Reset', self)
        self.reset.clicked.connect(self.reset_model)

        self.evaluate = qw.QPushButton('Evaluate', self)
        self.evaluate.clicked.connect(self.evaluate_model)

        self.tLayout = qw.QHBoxLayout(self)
        self.bLayout = qw.QHBoxLayout(self)
        self.tLayout.addWidget(self.tabs)
        self.bLayout.addWidget(self.reset)
        self.bLayout.addWidget(self.evaluate)
        self.layout.addLayout(self.tLayout)
        self.layout.addLayout(self.bLayout)
        self.setLayout(self.layout)


    def _xfspec_setup(self):
        layout = qw.QGridLayout(self)
        for i, xf in enumerate(self.parent.xfspec):
            js = range(len(xf))
            k = KeysTable(self, js)
            v = ValuesTable(self, [xf[j] for j in js])
            layout.addWidget(k, i, 1)
            layout.addWidget(v, i, 2)
            layout.addWidget(RowController(self, xf[0], k, v), i, 0)
        return layout

    def _modelspec_setup(self):
        layout = qw.QGridLayout(self)
        for i, m in enumerate(self.parent.ctx['modelspec'].fits()[0]):
            k = KeysTable(self, list(m['phi'].keys()))
            v = ValuesTable(self, list(m['phi'].values()))
            layout.addWidget(k, i, 1)
            layout.addWidget(v, i, 2)
            layout.addWidget(RowController(self, m['fn'], k, v), i, 0)
        return layout

    def reset_model(self):
        self.parent.ctx = copy.deepcopy(self.original_ctx)
        self.parent.xfspec = copy.deepcopy(self.original_xfspec)

        # TODO: Need to actually reset the values in the table as well.
        #       This approach is not working:
        self.xfspec_tab.layout = self._xfspec_setup()
        self.xfspec_tab.setLayout(self.xfspec_tab.layout)
        self.modelspec_tab.layout = self._modelspec_setup()
        self.modelspec_tab.setLayout(self.modelspec_tab.layout)

    def evaluate_model(self):
        xfspec, ctx = xforms.evaluate(self.parent.xfspec, self.parent.ctx,
                                      eval_model=True)
        self.parent.xfspec = xfspec
        self.parent.ctx = ctx
        self.parent.update_browser()

    def update_xfspec(self):
        # TODO: after a change is made to one of the entries,
        #       update in xfspec as well
        pass

    def update_modelspec(self):
        # TODO: after a change is made to one of the entries,
        #       update modelspec as well
        pass

class RowController(qw.QWidget):
    def __init__(self, parent, name, keys, values):
        super(qw.QWidget, self).__init__(parent)
        self.parent = parent
        self.name = name
        self.keys = keys
        self.values = values
        self.collapsed = True
        self.cols = []

        self.layout = qw.QHBoxLayout(self)
        self.toggle = qw.QPushButton('+/-', self)
        self.toggle.setFixedSize(40, 25)
        self.toggle.clicked.connect(self.toggle_collapsed)
        self.header = qw.QLabel(self.name, self)
        self.layout.addWidget(self.toggle, 0, qc.Qt.AlignTop)
        self.layout.addWidget(self.header, 0, qc.Qt.AlignTop)
        self.setLayout(self.layout)


    def toggle_collapsed(self):
        if self.collapsed:
            self._expand()
        else:
            self._collapse()
        self.collapsed = not self.collapsed

    def _collapse(self):
        self.keys._collapse()
        self.values._collapse()

    def _expand(self):
        self.values._expand()
        self.keys._expand()


class KeysTable(qw.QWidget):
    def __init__(self, parent, keys):
        super(qw.QWidget, self).__init__(parent)
        self.parent = parent
        self.keys = keys

        self.layout = qw.QVBoxLayout(self)
        for k in self.keys:
            self.layout.addWidget(qw.QLabel(str(k), self), 0, qc.Qt.AlignTop)
        self.setLayout(self.layout)
        self._collapse()

    def _collapse(self):
        for i in range(self.layout.count()):
            w = self.layout.itemAt(i).widget()
            w.hide()

    def _expand(self):
        for i in range(self.layout.count()):
            w = self.layout.itemAt(i).widget()
            w.show()


class ValuesTable(qw.QWidget):
    def __init__(self, parent, values):
        super(qw.QWidget, self).__init__(parent)
        self.parent = parent
        self.values = values

        self.layout = qw.QVBoxLayout(self)
        for v in self.values:
            if isinstance(v, np.ndarray) and not np.isscalar(v):
                self.layout.addWidget(ArrayModel(self, v), 0, qc.Qt.AlignTop)
            else:
                self.layout.addWidget(qw.QLineEdit(str(v), self), 0, qc.Qt.AlignTop)
        self.setLayout(self.layout)
        self._collapse()

    def _collapse(self):
        for i in range(self.layout.count()):
            w = self.layout.itemAt(i).widget()
            w.hide()

    def _expand(self):
        for i in range(self.layout.count()):
            w = self.layout.itemAt(i).widget()
            w.show()


# Just for testing - typically will be opened by recording_browser.py
# ctx and xfspec should be loaded into current console environment elsewhere
def run(ctx, xfspec):
    app = qw.QApplication(sys.argv)
    ex = ModelEditor(ctx=ctx, xfspec=xfspec)
    sys.exit(app.exec_())

if __name__ == '__main__':
    run(ctx, xfspec)
