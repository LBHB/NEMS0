import sys
import copy

import PyQt5.QtCore as qc
import PyQt5.QtGui as qg
import PyQt5.QtWidgets as qw

from nems import xforms


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
        table = qw.QTableWidget(self)
        table.setRowCount(len(spec))
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(['fn', 'kwargs', 'additional'])
        header = table.horizontalHeader()
        for i in range(3):
            header.setSectionResizeMode(i, qw.QHeaderView.ResizeToContents)
        for i, xf in enumerate(spec):
            table.setItem(i, 0, qw.QTableWidgetItem(xf[0]))

        layout.addWidget(table)
        return layout

    def _modelspec_setup(self):
        spec = self.parent.ctx['modelspecs'][0]
        layout = qw.QVBoxLayout(self)
        table = qw.QTableWidget(self)
        table.setRowCount(len(spec))
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(['fn', 'phi', 'value(s)'])
        header = table.horizontalHeader()
        for i in range(3):
            header.setSectionResizeMode(i, qw.QHeaderView.ResizeToContents)
        for i, m in enumerate(spec):
            table.setItem(i, 0, qw.QTableWidgetItem(m['fn']))

        layout.addWidget(table)
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


# Just for testing - typically will be opened by recording_browser.py
# ctx and xfspec should be loaded into current console environment elsewhere
def run(ctx, xfspec):
    app = qw.QApplication(sys.argv)
    ex = ModelEditor(ctx=ctx, xfspec=xfspec)
    sys.exit(app.exec_())

if __name__ == '__main__':
    run(ctx, xfspec)
