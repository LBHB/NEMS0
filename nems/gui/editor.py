import sys
import copy
import json
import logging

import numpy as np
import PyQt5.QtWidgets as qw
import PyQt5.QtCore as qc

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
        spec = self.parent.xfspec
        layout = qw.QVBoxLayout(self)
        for i, xf in enumerate(spec):
            name = xf[0]
            w = XformsRow(self, name, xf)
            layout.addWidget(w)

        return layout

    def _modelspec_setup(self):
        spec = self.parent.ctx['modelspecs'][0]
        layout = qw.QVBoxLayout(self)
        for i, m in enumerate(spec):
            name = m['fn']
            w = MspecRow(self, name, m)
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
        # TODO: after a change is made to one of the entries,
        #       update in xfspec as well
        pass

    def update_modelspec(self):
        # TODO: after a change is made to one of the entries,
        #       update modelspec as well
        pass


class RowWidget(qw.QWidget):
    def __init__(self, parent, name, contents):
        super(qw.QWidget, self).__init__(parent)
        self.parent = parent
        self.name = name
        self.contents = contents
        self.collapsed = True
        self.cols = []

        self.Vlayout = qw.QVBoxLayout(self)
        self.HlayoutTop = qw.QHBoxLayout(self)
        self.HlayoutBot = qw.QHBoxLayout(self)
        self.buttonsLayout = qw.QHBoxLayout(self)
        self.header = qw.QLabel(self.name, self)
        self.toggle = qw.QPushButton('+/-', self)
        self.toggle.setFixedSize(40, 25)
        self.toggle.clicked.connect(self.toggle_collapsed)

        self.HlayoutTop.addWidget(self.header)
        self.buttonsLayout.addWidget(self.toggle, 0, qc.Qt.AlignTop)
        self.HlayoutBot.addLayout(self.buttonsLayout)
        self._columns_setup()
        self.Vlayout.addLayout(self.HlayoutTop)
        self.Vlayout.addLayout(self.HlayoutBot)
        self.setLayout(self.Vlayout)

    def _columns_setup(self):
        pass

    def toggle_collapsed(self):
        if self.collapsed:
            self._expand()
        else:
            self._collapse()
        self.collapsed = not self.collapsed

    def _collapse(self):
        [c._collapse() for c in self.cols]

    def _expand(self):
        [c._expand() for c in self.cols]


class XformsRow(RowWidget):
    def _columns_setup(self):
        header1 = qw.QLabel('kwargs', self)
        col1 = ColumnWidget(self, *self.contents[1].keys())
        header2 = qw.QLabel('values', self)
        col2 = ColumnWidget(self, *self.contents[1].values())
        self.cols = [col1, col2]
        self.HlayoutTop.addWidget(header1)
        self.HlayoutBot.addWidget(col1, 0, qc.Qt.AlignTop)
        self.HlayoutTop.addWidget(header2)
        self.HlayoutBot.addWidget(col2, 0, qc.Qt.AlignTop)


class MspecRow(RowWidget):
    def _columns_setup(self):
        header1 = qw.QLabel('phi', self)
        col1 = ColumnWidget(self, *self.contents['phi'].keys())
        header2 = qw.QLabel('values', self)
        col2 = ColumnWidget(self, *self.contents['phi'].values())
        self.cols = [col1, col2]
        self.HlayoutTop.addWidget(header1)
        self.HlayoutBot.addWidget(col1, 0, qc.Qt.AlignTop)
        self.HlayoutTop.addWidget(header2)
        self.HlayoutBot.addWidget(col2, 0, qc.Qt.AlignTop)


class ColumnWidget(qw.QWidget):
    def __init__(self, parent, *items):
        super(qw.QWidget, self).__init__(parent)
        self.parent = parent
        self.items = items

        self.layout = qw.QVBoxLayout(self)
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
            # Just assume it's suitable as a string9cfilrv-
            return str(item)

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
