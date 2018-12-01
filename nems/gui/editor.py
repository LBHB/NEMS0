import sys
import copy
import json
import logging

import numpy as np
import PyQt5.QtWidgets as qw

from nems import xforms

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
        self.left = 0
        self.top = 0
        self.width = 600
        self.height = 800
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.editor = EditorWidget(self)
        self.setCentralWidget(self.editor)

        self.show()

    def update_browser(self):
        # TODO:
        pass


class EditorWidget(qw.QWidget):
    def __init__(self, parent):
        super(qw.QWidget, self).__init__(parent)
        self.parent = parent
        self.layout = qw.QVBoxLayout(self)
        self.tabs = qw.QTabWidget(self)
        self.xfspec_tab = qw.QWidget()
        self.modelspec_tab = qw.QWidget()
        self.tabs.resize(600, 800)
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
        spec = self.parent.xfspec
        layout = qw.QVBoxLayout(self)
        for i, xf in enumerate(spec):
            name = xf[0]
            col1 = ColumnWidget(self, 'kwargs', *list(xf[1].keys()))
            col2 = ColumnWidget(self, 'values', *list(xf[1].values()))
            w = RowWidget(self, name, col1, col2)
            layout.addWidget(w)
        return layout

    def _modelspec_setup(self):
        spec = self.parent.ctx['modelspecs'][0]
        layout = qw.QVBoxLayout(self)
        for i, m in enumerate(spec):
            name = m['fn']
            col1 = ColumnWidget(self, 'phi', *list(m['phi'].keys()))
            col2 = ColumnWidget(self, 'values', *list(m['phi'].values()))
            w = RowWidget(self, name, col1, col2)
            layout.addWidget(w)

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
        # TODO
        pass

    def update_modelspec(self):
        # TODO
        pass


class RowWidget(qw.QWidget):
    def __init__(self, parent, name, *items):
        super(qw.QWidget, self).__init__(parent)
        self.parent = parent
        self.name = name
        self.items = items
        self.collapsed = True

        self.layout = qw.QHBoxLayout(self)
        self.header = qw.QLabel(self.name, self)
        self.toggle = qw.QPushButton('+/-', self)
        self.toggle.clicked.connect(self.toggle_collapsed)
        self.layout.addWidget(self.header)
        self.layout.addWidget(self.toggle)
        for it in self.items:
            self.layout.addWidget(it)
        self.setLayout(self.layout)

    def toggle_collapsed(self):
        if self.collapsed:
            self._expand()
        else:
            self._collapse()
        self.collapsed = not self.collapsed

    def _collapse(self):
        [it._collapse() for it in self.items]

    def _expand(self):
        [it._expand() for it in self.items]


class ColumnWidget(qw.QWidget):
    def __init__(self, parent, name, *items):
        super(qw.QWidget, self).__init__(parent)
        self.parent = parent
        self.name = name
        self.items = items

        self.layout = qw.QVBoxLayout(self)
        self.header = qw.QLabel(self.name, self)
        self.layout.addWidget(self.header)
        for i, it in enumerate(self.items):
            contents = self._parse_item(it, i)
            line = qw.QLineEdit(contents, self)
            line.setVisible(False)
            self.layout.addWidget(line)
        self.setLayout(self.layout)

    def _parse_item(self, item, idx):
        if isinstance(item, np.ndarray):
            # TODO
            return 'array'

        else:
            # Just assume it's suitable as a string
            return str(item)

    def _collapse(self):
        for i in range(self.layout.count())[1:]:
            w = self.layout.itemAt(i).widget()
            w.hide()

    def _expand(self):
        for i in range(self.layout.count())[1:]:
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
