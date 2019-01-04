#import matplotlib
#matplotlib.use('Qt5Agg')
import sys
import copy
import json
import logging

import numpy as np
import PyQt5.QtWidgets as qw
import PyQt5.QtCore as qc
import PyQt5.QtGui as qg

from nems import xforms
from nems.gui.models import ArrayModel
from nems.gui.canvas import MyMplCanvas
import nems.db as nd

log = logging.getLogger(__name__)

_SCROLLABLE_PLOT_FNS = [
        'nems.plots.api.strf_timeseries',
        'nems.plots.api.before_and_after',
        'nems.plots.api.pred_resp',
        ]

# TODO: redo some of these functions to take advantage of new modelspec setup?

# TODO: implement some kind of coordinate system to keep track of where changes
#       to individual phi etc need to be propagated


class EditorWindow(qw.QMainWindow):
    def __init__(self, modelspec=None, xfspec=None, rec=None, ctx=None,
                 rec_name='val'):
        super(qw.QMainWindow, self).__init__()
        self.title = 'NEMS Model Browser'
        if (modelspec is None) and (ctx is not None):
            modelspec = ctx.get('modelspec', None)
        if (rec is None) and (ctx is not None):
            rec = ctx.get(rec_name, None)
        self.editor = EditorWidget(modelspec, xfspec, rec, ctx, self)
        self.setCentralWidget(self.editor)
        self.setWindowTitle(self.title)
        self.show()


class EditorWidget(qw.QWidget):
    def __init__(self, modelspec=None, xfspec=None, rec=None, ctx=None,
                 parent=None):
        super(qw.QWidget, self).__init__()
        self.xfspec = xfspec
        self.modelspec = modelspec
        self.rec = rec
        if ctx is None:
            self.ctx = {}
        else:
            self.ctx = ctx

        self.xfspec = xfspec
        self.modelspec = modelspec
        self.rec = rec
        self.title = 'NEMS Model Browser'

        outer_layout = qw.QVBoxLayout()
        row_one_layout = qw.QHBoxLayout()
        row_two_layout = qw.QHBoxLayout()

        if (modelspec is not None) and (rec is not None):
            self.modelspec.recording = rec
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

        layout = qw.QGridLayout()
        self.modules = [ModuleEditor(i, m, self)
                        for i, m in enumerate(self.modelspec.modules)]
        self.controllers = [ModuleControls(m, self) for m in self.modules]
        self.collapsers = [ModuleCollapser(m, self) for m in self.modules]

        i = 0
        widgets = zip(self.collapsers, self.controllers, self.modules)
        for i, (col, cnt, m) in enumerate(widgets):
            layout.addWidget(col, i, 0)
            layout.addWidget(cnt, i, 1)
            layout.addWidget(m, i, 2)

        layout.setAlignment(qc.Qt.AlignTop)
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

    def evaluate_model(self, first_changed_module=0):
        # TODO: Fix issues with first_changed_module.
        new_rec = self.parent.modelspec.evaluate()#start=first_changed_module)
        self.parent.modelspec.recording = new_rec
        #for m in self.modules[first_changed_module:]:
        for m in self.modules:
            m.new_plot()


class ModuleEditor(qw.QWidget):
    def __init__(self, mod_index, data, parent):
        super(qw.QWidget, self).__init__()
        self.mod_index = mod_index
        self.parent = parent

        # Default plot options - set them up here then change w/ controller
        self.plot_fn_idx = data.get('plot_fn_idx', 0)
        self.fit_index = parent.modelspec.fit_index
        # TODO: Need to do something smarter with this
        self.sig_name = 'pred'
        self.scrollable = self.check_scrollable()

        self.layout = qw.QHBoxLayout()
        self.canvas = MyMplCanvas(parent=self)
        self.layout.addWidget(self.canvas)
        self.layout.setAlignment(qc.Qt.AlignTop)
        self.setLayout(self.layout)

        # Draw initial plot
        self.plot_on_axes()

    def new_plot(self):
        self.layout.removeWidget(self.canvas)
        self.canvas.close()
        self.canvas = MyMplCanvas(parent=self)
        self.plot_on_axes()
        self.layout.addWidget(self.canvas)
        self.scrollable = self.check_scrollable()
        self.update_plot()

    def plot_on_axes(self):
        ax = self.canvas.figure.add_subplot(111)
        rec = self.parent.modelspec.recording
        self.parent.modelspec.plot(self.mod_index, rec, ax,
                                   self.plot_fn_idx, self.fit_index,
                                   self.sig_name, no_legend=True)
        self.canvas.draw()

    def check_scrollable(self):
        plots = self.parent.modelspec[self.mod_index].get(
                'plot_fns', ['nems.plots.api.mod_output']
                )
        if plots[self.plot_fn_idx] in _SCROLLABLE_PLOT_FNS:
            self.scrollable = True
        else:
            self.scrollable = False

    def update_plot(self):
        if self.scrollable:
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
        else:
            pass


class ModuleCollapser(qw.QWidget):
    def __init__(self, module, parent):
        super(qw.QWidget, self).__init__()
        self.module = module
        self.controller = parent.controllers[module.mod_index]
        self.parent = parent
        self.collapsed = False

        layout = qw.QVBoxLayout()
        self.toggle = qw.QPushButton('+/-', self)
        self.toggle.setFixedSize(40, 25)
        self.toggle.clicked.connect(self.toggle_collapsed)
        layout.addWidget(self.toggle)
        layout.setAlignment(qc.Qt.AlignTop)
        self.setLayout(layout)

    def toggle_collapsed(self):
        if self.collapsed:
            self.module.show()
            self.controller.show()
        else:
            self.module.hide()
            self.controller.hide()
        self.collapsed = not self.collapsed


class ModuleControls(qw.QWidget):
    def __init__(self, module, parent=None):
        super(qw.QWidget, self).__init__()
        self.module = module
        self.parent = parent
        self.mod_index = self.module.mod_index
        self.module_data = copy.deepcopy(
                self.module.parent.modelspec[self.mod_index]
                )

        layout = qw.QVBoxLayout()

        plot_list = self.module_data.get('plot_fns', [])
        self.plot_functions_menu = qw.QComboBox()
        self.plot_functions_menu.addItems(plot_list)
        initial_index = self.module.plot_fn_idx
        if initial_index is None:
            initial_index = 0
        self.plot_functions_menu.setCurrentIndex(initial_index)
        layout.addWidget(self.plot_functions_menu)
        self.plot_functions_menu.currentIndexChanged.connect(self.change_plot)

        self.edit_phi = qw.QPushButton('Edit Phi')
        self.edit_phi.clicked.connect(self.open_phi)
        layout.addWidget(self.edit_phi)

        self.phi_editor = PhiEditor(self.module_data['phi'], self)
        self.phi_editor.hide()
        self.save_phi = qw.QPushButton('Save Phi')
        self.save_phi.hide()
        self.save_phi.clicked.connect(self.close_phi)
        layout.addWidget(self.phi_editor)
        layout.addWidget(self.save_phi)

        layout.setAlignment(qc.Qt.AlignTop)
        self.setLayout(layout)

    def change_plot(self, index):
        self.module.plot_fn_idx = int(index)
        self.module.new_plot()

    def open_phi(self):
        self.phi_editor.show()
        self.save_phi.show()

    def close_phi(self):
        new_phi = self.phi_editor.export_phi()
        if not self.phi_equal(new_phi):
            need_evaluate = True
        else:
            need_evaluate = False

        self.parent.modelspec[self.mod_index]['phi'] = copy.deepcopy(new_phi)
        self.phi_editor.hide()
        self.save_phi.hide()

        if need_evaluate:
            self.parent.evaluate_model(first_changed_module=self.mod_index)
            self.module_data['phi'] = copy.deepcopy(new_phi)

    def phi_equal(self, phi2):
        equal = True
        phi1 = self.parent.modelspec[self.mod_index]['phi']
        for v1, v2 in zip(phi1.values(), phi2.values()):
            if not np.array_equal(v1, v2):
                equal = False
                break
        return equal


class PhiEditor(qw.QWidget):
    def __init__(self, phi, parent):
        super(qw.QWidget, self).__init__(parent)
        self.phi = phi
        self.original_phi = copy.deepcopy(phi)
        self.parent = parent

        layout = qw.QFormLayout()
        self.arrays = {}
        for k, v in phi.items():
            label = qw.QLabel(k)
            array = ArrayModel(self, v)
            self.arrays[k] = array
            layout.addRow(label, array)

        layout.setAlignment(qc.Qt.AlignTop)
        self.setLayout(layout)

    def export_phi(self):
        return {k: v.array for k, v in self.arrays.items()}


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
        raise NotImplementedError

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

    def filtered_xfspec(self):
        checks = [s.checked for s in self.steps]
        x = [s for s, c in zip(self.xfspec, checks) if c]
        return x


class XfStepEditor(qw.QWidget):
    def __init__(self, index, step, parent):
        super(qw.QWidget, self).__init__()
        self.index = index
        self.step = step
        self.parent = parent
        self.checked = True

        # need to be able to turn steps on and off
        # need to be able to get and set values from the step (which should
        # be a list of either 2 or 4 items)
        # need to propagate updates back to parent xfspec

        layout = qw.QVBoxLayout()
        self.checkbox = qw.QCheckBox(self.step[0], self)
        self.checkbox.setCheckState(qc.Qt.Checked)
        self.checkbox.stateChanged.connect(self.toggle)
        layout.addWidget(self.checkbox)
        self.setLayout(layout)

    def toggle(self):
        self.checked = not self.checked


class GlobalControls(qw.QWidget):
    start_time = 0
    display_duration = 10.0
    minimum_duration = 0.001
    stop_time = 10

    def __init__(self, modelspec, parent):
        super(qw.QWidget, self).__init__()
        self.modelspec = modelspec
        self.parent = parent

        # Slider for plot view windows
        self._update_max_time()
        self.time_slider = qw.QScrollBar(orientation=1)
        self.time_slider.setRange(0, self.max_time-self.display_duration)
        self.time_slider.setRepeatAction(200, 2)
        self.time_slider.setSingleStep(1)
        self.time_slider.valueChanged.connect(self.scroll_all)

        # Set zoom / display range for plot views
        self.display_range = qw.QLineEdit()
        self.display_range.setValidator(
                qg.QDoubleValidator(self.minimum_duration, 10000.0, 4)
                )
        self.display_range.editingFinished.connect(self.set_display_range)
        self.display_range.setText(str(self.display_duration))

        # Increment / Decrement zoom
        plus = qw.QPushButton('Zoom Out')
        plus.clicked.connect(self.increment_display_range)
        minus = qw.QPushButton('Zoom In')
        minus.clicked.connect(self.decrement_display_range)
        range_layout = qw.QHBoxLayout()
        [range_layout.addWidget(w) for w in [self.display_range, plus, minus]]

        layout = qw.QVBoxLayout()
        layout.addWidget(self.time_slider)
        layout.addLayout(range_layout)
        self.setLayout(layout)

        #self._update_range()

    # Plot window adjustments
    def scroll_all(self):
        self.start_time = self.time_slider.value()
        self.stop_time = self.start_time + self.display_duration

        # don't go past the latest time of the biggest plot
        # (should all have the same max most of the time)
        self._update_max_time()
        if self.stop_time >= self.max_time:
            self.stop_time = self.max_time
            self.start_time = max(0, self.max_time - self.display_duration)

        [m.update_plot() for m in self.parent.modelspec_editor.modules]

    def _update_max_time(self):
        resp = self.parent.rec['resp']
        self.max_time = resp.as_continuous().shape[-1] / resp.fs


    def tap_right(self):
        self.time_slider.set_value(
                self.time_slider.value + self.time_slider.singleStep
                )

    def tap_left(self):
        self.time_slider.set_value(
                self.time_slider.value - self.time_slider.singleStep
                )

    def set_display_range(self):
        duration = float(self.display_range.text())
        if not duration:
            print("Duration not set to a valid value. Please enter a"
                  "a number > 0")
            return
        self.display_duration = duration
        self._update_range()

    def increment_display_range(self):
        self.display_duration += 1
        self.display_range.setText(str(self.display_duration))
        self._update_range()

    def decrement_display_range(self):
        self.display_duration -= 1
        self.display_range.setText(str(self.display_duration))
        self._update_range()

    def _update_range(self):
        self.time_slider.setRange(0, self.max_time-self.display_duration)
        self.time_slider.setSingleStep(int(np.ceil(self.display_duration/10)))
        self.time_slider.setPageStep(int(self.display_duration))
        self.scroll_all()


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
def run(modelspec, xfspec, rec, ctx):
    app = qw.QApplication(sys.argv)
    ex = EditorWindow(modelspec=modelspec, xfspec=xfspec, rec=rec, ctx=ctx)
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

    if 'load' in sys.argv:
        batch = 271
        cellid = "TAR010c-18-1"
        modelname = 'dlog-wc.18x1.g-fir.1x15-lvl.1'
        d = nd.get_results_file(batch, modelnames=[modelname], cellids=[cellid])
        filename = d.loc[0,'modelpath']
        xfspec, ctx = xforms.load_analysis(filename)
        #xfspec, ctx = xforms.load_analysis(
        #     "/home/jacob/code/nems/results/271/TAR010c-18-1/"
        #     "TAR010c.dlog_wc.18x1.g_fir.1x15_lvl.1_dexp.1.fit_basic"
        #     ".2018-12-24T014712/"
        #     )
        modelspec = ctx['modelspec']
        rec = ctx['val']

    run(modelspec, xfspec, rec, ctx)
