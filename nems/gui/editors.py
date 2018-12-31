import sys
import copy
import json
import logging

import numpy as np
import PyQt5.QtWidgets as qw
import PyQt5.QtCore as qc

from nems import xforms
from nems.gui.models import ArrayModel
from nems.gui.canvas import MyMplCanvas

log = logging.getLogger(__name__)


# TODO: redo some of these functions to take advantage of new modelspec setup?

# TODO: implement some kind of coordinate system to keep track of where changes
#       to individual phi etc need to be propagated


class EditorWindow(qw.QMainWindow):
    def __init__(self, modelspec=None, xfspec=None, rec=None):
        super(qw.QMainWindow, self).__init__()
        self.title = 'NEMS Model Browser'
        self.editor = EditorWidget(modelspec, xfspec, rec, self)
        self.setCentralWidget(self.editor)
        self.setWindowTitle(self.title)
        self.show()


class EditorWidget(qw.QWidget):
    def __init__(self, modelspec=None, xfspec=None, rec=None, parent=None):
        super(qw.QWidget, self).__init__()
        self.xfspec = xfspec
        self.modelspec = modelspec
        self.rec = rec

        self.xfspec = xfspec
        self.modelspec = modelspec
        self.rec = rec
        self.title = 'NEMS Model Browser'

        outer_layout = qw.QVBoxLayout()
        row_one_layout = qw.QHBoxLayout()
        row_two_layout = qw.QHBoxLayout()

        if (modelspec is not None) and (rec is not None):
            self.modelspec_editor = ModelspecEditor(modelspec, rec, self)
        if self.xfspec is not None:
            self.xfspec_editor = XfspecEditor(self.xfspec, self)

        self.global_controls = GlobalControls(self.modelspec, self)
        self.fit_editor = FitEditor(self.modelspec, self.xfspec, self)

        row_one_layout.addWidget(self.modelspec_editor)
        row_one_layout.addWidget(self.xfspec_editor)
        row_two_layout.addWidget(self.global_controls)
        row_two_layout.addWidget(self.fit_editor)
        outer_layout.addLayout(row_one_layout)
        outer_layout.addLayout(row_two_layout)
        self.setLayout(outer_layout)

        self.setWindowTitle(self.title)
        self.show()


class ModelspecEditor(qw.QWidget):
    def __init__(self, modelspec, rec, parent=None):
        super(qw.QWidget, self).__init__()
        self.modelspec = modelspec
        self.original_modelspec = copy.deepcopy(modelspec)
        self.rec = rec
        self.parent = parent

        layout = qw.QVBoxLayout()
        self.modules = [ModuleEditor(i, m, self)
                        for i, m in enumerate(self.modelspec)]
        self.collapsers = [ModuleCollapser(m, self) for m in self.modules]
        for m, c in zip(self.modules, self.collapsers):
            row = qw.QHBoxLayout()
            row.addWidget(c)
            row.addWidget(m)
            layout.addLayout(row)

        #module_layout = qw.QVBoxLayout()
        #collapser_layout = qw.QVBoxLayout()
        #[module_layout.addWidget(m) for m in self.modules]
        #[collapser_layout.addWidget(c) for c in self.collapsers]
        #outer_layout.addLayout(collapser_layout)
        #outer_layout.addLayout(module_layout)
        self.setLayout(layout)

    def update_modelspec(self):
        raise NotImplementedError

        phis = []
        for w in self.modelspec_tab.values:
            p = {}
            for k, v in zip(w.keys, w.values):
                p[k] = v
            phis.append(p)

        # TODO: just use some method of modelspec object to update phi instead
        modelspec = copy.deepcopy(self.parent.ctx['modelspec'].raw)
        for i, p in enumerate(phis):
            modelspec[i]['phi'] = p
        self.parent.ctx['modelspec'].raw = modelspec

    def evaluate_model(self):
        raise NotImplementedError

        # Make sure xfspec and modelspec are up to date before evaluating
        self.update_xfspec()
        self.update_modelspec()
        xfspec, ctx = xforms.evaluate(self.parent.xfspec, self.parent.ctx,
                                      eval_model=True)
        self.parent.xfspec = xfspec
        self.parent.ctx = ctx
        self.parent.update_browser()


class ModuleEditor(qw.QWidget):
    def __init__(self, mod_index, module, parent):
        super(qw.QWidget, self).__init__()
        self.mod_index = mod_index
        self.module = module
        self.parent = parent

        # Default plot options - set them up here then change w/ controller
        self.plot_fn_idx = None
        self.fit_index = None
        self.sig_name = 'pred'

        layout = qw.QHBoxLayout()
        controls_layout = qw.QVBoxLayout()
        plot_layout = qw.QVBoxLayout()

        self.controls = ModuleControls(self.module, parent=self)
        controls_layout.addWidget(self.controls)
        self.canvas = MyMplCanvas(parent=self)
        plot_layout.addWidget(self.canvas)

        layout.addLayout(controls_layout)
        layout.addLayout(plot_layout)
        self.setLayout(layout)

        # Draw initial plot
        self.new_plot()

    def new_plot(self):
        self.axes = self.canvas.figure.add_subplot(111)
        self.parent.modelspec.plot(self.mod_index, self.parent.rec, self.axes,
                                   self.plot_fn_idx, self.fit_index,
                                   self.sig_name)
        self.canvas.draw()

    def update_plot(self):
        gc = self.parent.parent.global_controls
        try:
            fs = self.parent.rec[self.sig_name].fs
        except AttributeError:
            log.warning('No sampling rate for signal: %s' % self.sig_name)
            fs = 1

        self.canvas.axes.set_xlim(gc.start_time*fs, gc.stop_time*fs)
#        if not (self.point or self.tiled):
#            self.axes.set_ylim(ymin=self.ymin, ymax=self.ymax)
        self.canvas.draw()


class ModuleCollapser(qw.QWidget):
    def __init__(self, module, parent):
        super(qw.QWidget, self).__init__()
        self.module = module
        self.parent = parent
        self.collapsed = False

        layout = qw.QVBoxLayout()
        self.toggle = qw.QPushButton('+/-', self)
        self.toggle.setFixedSize(40, 25)
        self.toggle.clicked.connect(self.toggle_collapsed)
        layout.addWidget(self.toggle)
        self.setLayout(layout)

    def toggle_collapsed(self):
        if self.collapsed:
            self.module.show()
        else:
            self.module.hide()
        self.collapsed = not self.collapsed


class ModuleControls(qw.QWidget):
    def __init__(self, module, parent=None):
        super(qw.QWidget, self).__init__()
        self.module = module
        self.parent = parent

        layout = qw.QVBoxLayout()
        self.test = qw.QLineEdit('test module controls')
        layout.addWidget(self.test)
        self.setLayout(layout)


class XfspecEditor(qw.QWidget):
    def __init__(self, xfspec, parent=None):
        super(qw.QWidget, self).__init__()
        self.xfspec = xfspec
        self.original_xfspec = copy.deepcopy(xfspec)
        self.parent = parent

        self.steps = [XfStepEditor(i, s, self)
                      for i, s in enumerate(self.xfspec)]
        self.step_layout = qw.QVBoxLayout()
        [self.step_layout.addWidget(s) for s in self.steps]
        self.setLayout(self.step_layout)

    def update_xfspec(self):
        xfspec = []
        for w in self.xfspec_tab.values:
            xf = []
            for k, v in zip(w.keys, w.values):
                try:
                    v = json.loads(v)
                except TypeError:
                    # Want to un-string dictionaries etc, but not ndarrays
                    pass
                xf.append(v)
            xfspec.append(xf)

        self.parent.xfspec = xfspec


class XfStepEditor(qw.QWidget):
    def __init__(self, index, step, parent):
        super(qw.QWidget, self).__init__()
        self.index = index
        self.step = step
        self.parent = parent

        # need to be able to turn steps on and off
        # need to be able to get and set values from the step (which should
        # be a list of either 2 or 4 items)
        # need to propagate updates back to parent xfspec

        layout = qw.QVBoxLayout()
        self.test = qw.QLineEdit('test xfstep editor')
        layout.addWidget(self.test)
        self.setLayout(layout)


class GlobalControls(qw.QWidget):
    def __init__(self, modelspec, parent):
        super(qw.QWidget, self).__init__()
        self.modelspec = modelspec
        self.parent = parent

        # update fit and cell index for parent modelspec
        # control time scroll for scrollable plots

        layout = qw.QVBoxLayout()
        self.test = qw.QLineEdit('test global controls')
        layout.addWidget(self.test)
        self.setLayout(layout)


class FitEditor(qw.QWidget):
    def __init__(self, modelspec, xfspec, parent):
        super(qw.QWidget, self).__init__()
        self.modelspec = modelspec
        self.xfspec = xfspec
        self.parent = parent

        # Be able to pick out fitter steps from xfspec?
        # Or maybe just pre-specifify the fitter functions with
        # an option to look up additional ones

        # this is per SVD request separate from xfspec editor
        # want to be able to easily switch between different fits,
        # do only initialization, etc.

        layout = qw.QVBoxLayout()
        self.test = qw.QLineEdit('test fit editor')
        layout.addWidget(self.test)
        self.setLayout(layout)


# Just for testing - typically will be opened by recording_browser.py
# ctx and xfspec should be loaded into current console environment elsewhere
def run(modelspec, xfspec, rec):
    app = qw.QApplication(sys.argv)
    ex = EditorWindow(modelspec=modelspec, xfspec=xfspec, rec=rec)
    sys.exit(app.exec_())


_DEBUG = False
if __name__ == '__main__':
    if _DEBUG:
        sys._excepthook = sys.excepthook
        def exception_hook(exctype, value, traceback):
            print(exctype, value, traceback)
            sys._excepthook(exctype, value, traceback)
            sys.exit(1)
        sys.excepthook = exception_hook
    run(modelspec, xfspec, rec)
