'''
Contains relevant classes for contsructing the NEMS model editor.

The different parts of the editor are layed out as follows:

------------------------------EditorWindow--------------------------------------
: :------------------------------ EditorWidget--------------------------------::
: : :-------------------ModelspecEditor-----------------: :---XfspecEditor---:::
: : : EpochsCollapser |                  | EpochsCanvas : :                  :::
: : :                                                   : :                  :::
: : : ModuleCollapser |-ModuleController-| ModuleCanvas : :   XfstepEditor   :::
: : :      [+/-]        :  PhiEditor   :      [Plot]    : :                  :::
: : :                   :..............:                : :                  :::
: : :          (1 of each per module in modelspec)      : :  (1 per step in  :::
: : :                                                   : :     the xfspec)  :::
: : :                         ...                       : :        ...       :::
: : :                                                   : :                  :::
: : :...................................................: :..................:::
: :                                                                           ::
: : ---------GlobalControls-----------------------------: :------FitEditor---:::
: : :  <----------------Plot Scrollbar--------------->  : :[Init Fn][Do Init]:::
: : : [Plot Display Duration]    [Zoom Out]   [Zoom In] : :[Fit Fn] [#] [Fit]:::
: : : [Reset Modelspec] [Set Fit index] [Set Cell Index]: :                  :::
: : :...................................................: :..................:::
: :...........................................................................::
:..............................................................................:

There are also buttons along the left, right, and bottom edges for toggling
the visibility of the ModelspecEditor, XfspecEditor, and the GlobalControls
& FitEditor, respectively.

Most modules will only need to refer to their immediate parent. However,
ModuleCollapsers reference the ModuleController and ModuleCanvas within
the same row in order to toggle their visibility. Additionally, GlobalControls
needs to reference many distantly related modules.

Classes
-------
EditorWindow(QMainWindow): Unpacks ctx if necessary and contains EditorWidget.
EditorWidget(QWidget): Keeps a reference to all data and lays out the
                       ModelspecEditor, XfspecEditor, GlobalControls, and
                       FitEditor.
ModelspecEditor(QWidget): Handles evaluation of modelspec and lays out a
                          ModuleCollapser, ModuleController, and ModuleCanvas
                          for each module in the modelspec.
ModuleCollapser(QWidget): Toggles visibility of one module.
ModuleController(QWidget): Controls for changing plot type, editing phi, etc.
                           for one module.
ModuleCanvas(QWidget): Plot display for one module.
EpochsCollapser(QWidget): Toggles visibilty of the epochs display.
EpochsWrapper(QWidget): Contains the EpochCanvas. Simple layout to maintain
                        consistency of appearance with module canvases.
XfspecEditor(QWidget): Handles evaluation of xfspec and lays out an XfstepEditor
                       for each step in the xfspec. Hidden by default.
                       (more features to be added later).
GlobalControls(QWidget): Provides user controls for adjusting plot limits and
                         changing fit and cell indices for modelspec
                         (more features to be added later)
FitEditor(QWidget): Provides user controls for re-initializing the modelspec
                    as well as incremental fitting
                    (more features to be added later).

'''

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
from nems.gui.canvas import NemsCanvas, EpochCanvas
from nems.modelspec import _lookup_fn_at
import nems.db as nd

log = logging.getLogger(__name__)

# Only module plots included here will be scrolled in time
# by the slider.
_SCROLLABLE_PLOT_FNS = [
    #'nems.plots.api.strf_timeseries',
    'nems.plots.api.state_vars_timeseries',
    'nems.plots.api.before_and_after',
    'nems.plots.api.pred_resp',
    'nems.plots.api.spectrogram_output',
    'nems.plots.api.spectrogram',
    'nems.plots.api.pred_spectrogram',
    'nems.plots.api.resp_spectrogram',
    'nems.plots.api.mod_output',
    'nems.plots.api.mod_output_all',
    'nems.plots.api.fir_output_all'
]

# These are used as click-once operations
# TODO: separate initialization from prefitting
_INIT_FNS = [
        'nems.initializers.from_keywords',
        'nems.initializers.prefit_LN'
        ]

# These can be repeated as needed in small steps
_FIT_FNS = [
        'nems.analysis.fit_basic.fit_basic',
        'nems.analysis.fit_iteratively.fit_iteratively'
        ]

# TODO: add backwards compatibility shim to add plot_fns, plot_fn_idx etc to
#       old modelspecs if none of the modules have those specified.

# TODO: Switch modelspec, xfspec etc. references to all just point to
#       EditorWidget copy instead of making separate copies.
#       Then all updates can use the most convenient
#       pointer instead of needing to call parent.parent.parent.modelspec



class EditorWindow(qw.QMainWindow):

    def __init__(self, modelspec=None, xfspec=None, rec=None, ctx=None,
                 rec_name='val'):
        '''
        Main Window wrapper for NEMS model editor GUI.
        Allows browsing and editing of fitted model parameters,
        xforms spec options (TODO), plotting data on a per-module
        basis, and manual initialization & fitting with adjustable
        iteration counts.

        Parameters
        ----------
        modelspec : ModelSpec
            A NEMS ModelSpec containing at least one module.
        xfspec : nested list
            A NEMS xforms spec (see nems.xforms) containing at least one step.
        rec : Recording
            A NEMS Recording, generally expected to contain 'stim', 'resp',
            and 'pred' signals.
        ctx : dict
            A NEMS context dictionary (see nems.xforms)
        rec_name : str
            Key used to set rec from ctx instead of passing rec directly,
            e.x. 'val' or 'est'.


        '''
        super(qw.QMainWindow, self).__init__()
        if (modelspec is None) and (ctx is not None):
            modelspec = ctx.get('modelspec', None)
        if (rec is None) and (ctx is not None):
            rec = ctx.get(rec_name, None)
        self.title = modelspec.meta['modelname']
        self.editor = EditorWidget(modelspec, xfspec, rec, ctx, self)
        self.setCentralWidget(self.editor)
        self.setWindowTitle(self.title)
        self.show()


class EditorWidget(qw.QWidget):

    def __init__(self, modelspec=None, xfspec=None, rec=None, ctx=None,
                 parent=None):
        '''
        Parameters
        ----------
        modelspec : ModelSpec
            A NEMS ModelSpec containing at least one module.
        xfspec : nested list
            A NEMS xforms spec (see nems.xforms) containing at least one step.
        rec : Recording
            A NEMS Recording, generally expected to contain 'stim', 'resp',
            and 'pred' signals.
        ctx : dict
            A NEMS context dictionary (see nems.xforms).
        parent : QWidget*
            Expected to be an instance of EditorWindow.

        '''
        super(qw.QWidget, self).__init__()
        self.xfspec = xfspec
        self.modelspec = modelspec
        self.rec = rec
        self.rec=self.rec.apply_mask(reset_epochs=True)
        self.modelspec.recording=self.rec
        if ctx is None:
            self.ctx = {}
        else:
            self.ctx = ctx

        self.title = 'NEMS Model Browser'
        # By default, start with xfspec steps hidden but all other
        # controls showing.
        self.modules_collapsed = False
        self.xfsteps_collapsed = True
        self.bottom_collapsed = False

        # Layout is split up into:
        # First row: ModelspecEditor (left), XfspecEditor (right)
        # Second row: GlobalControls (left), FitEditor (right)
        # Third row: Bottom collapser
        outer_layout = qw.QVBoxLayout()
        row_one_layout = qw.QHBoxLayout()
        row_two_layout = qw.QHBoxLayout()
        row_three_layout = qw.QHBoxLayout()

        self.modelspec_editor = ModelspecEditor(modelspec, self.rec, self)
        if self.xfspec is not None:
            self.xfspec_editor = XfspecEditor(self.xfspec, self)
        self.global_controls = GlobalControls(self)
        self.fit_editor = FitEditor(self)

        self.modelspec_editor.setup_layout()
        # Have to set up these plots afterward to get
        # canvases to fill the layout properly.
        self.modelspec_editor.refresh_plots()
        self.modelspec_editor.epochs.setup_figure()

        self.setup_module_collapser()
        self.setup_xfstep_collapser()
        self.hide_xfstep_controls()
        self.setup_bottom_collapser()

        row_one_layout.addLayout(self.module_collapser_layout)
        row_one_layout.addWidget(self.modelspec_editor)
        row_one_layout.addWidget(self.xfspec_editor)
        row_one_layout.addLayout(self.xfstep_collapser_layout)
        row_two_layout.addWidget(self.global_controls)
        row_two_layout.addWidget(self.fit_editor)
        row_two_layout.setContentsMargins(10, 10, 10, 2)
        outer_layout.addLayout(row_one_layout)
        outer_layout.addLayout(row_two_layout)
        outer_layout.addLayout(self.bottom_collapser_layout)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(outer_layout)

        self.setWindowTitle(self.title)
        self.show()

    def set_new_modelspec(self, new):
        '''Update modelspec references to new and evaluate it.'''
        self.modelspec = new
        self.ctx['modelspec'] = new
        self.modelspec.recording = self.rec
        self.modelspec_editor.modelspec = new
        self.modelspec_editor.modelspec.recording = self.rec
        self.modelspec_editor.evaluate_model()

    def setup_module_collapser(self):
        '''Add button to left of ModelspecEditor for hiding module controls.'''
        self.module_collapser_layout = qw.QVBoxLayout()

        self.module_collapser = qw.QToolButton(self)
        self.module_collapser.setMaximumWidth(15)
        self.module_collapser.clicked.connect(self.toggle_module_controls)
        self.module_collapser.setArrowType(qc.Qt.LeftArrow)
        policy = qw.QSizePolicy()
        policy.setVerticalPolicy(qw.QSizePolicy.Expanding)
        self.module_collapser.setSizePolicy(policy)

        self.module_collapser_layout.addWidget(self.module_collapser)

    def setup_xfstep_collapser(self):
        '''Add button to right of XfspecEditor for hiding xfstep controls.'''
        self.xfstep_collapser_layout = qw.QVBoxLayout()

        self.xfstep_collapser = qw.QToolButton(self)
        self.xfstep_collapser.setMaximumWidth(15)
        self.xfstep_collapser.clicked.connect(self.toggle_xfstep_controls)
        self.xfstep_collapser.setArrowType(qc.Qt.RightArrow)
        policy = qw.QSizePolicy()
        policy.setVerticalPolicy(qw.QSizePolicy.Expanding)
        self.xfstep_collapser.setSizePolicy(policy)

        self.xfstep_collapser_layout.addWidget(self.xfstep_collapser)

    def setup_bottom_collapser(self):
        '''Add button to bottom of window for hiding Global and Fit controls.'''
        self.bottom_collapser_layout = qw.QHBoxLayout()

        self.bottom_collapser = qw.QToolButton(self)
        self.bottom_collapser.setMaximumHeight(15)
        self.bottom_collapser.clicked.connect(self.toggle_bottom_controls)
        self.bottom_collapser.setArrowType(qc.Qt.DownArrow)
        policy = qw.QSizePolicy()
        policy.setHorizontalPolicy(qw.QSizePolicy.Expanding)
        self.bottom_collapser.setSizePolicy(policy)

        self.bottom_collapser_layout.addWidget(self.bottom_collapser)

    def toggle_module_controls(self):
        '''Hide module controls if visible, or show them if hidden.'''
        if self.modules_collapsed:
            self.show_module_controls()
        else:
            self.hide_module_controls()
        self.modules_collapsed = not self.modules_collapsed

    def hide_module_controls(self):
        '''Make module controls invisible.'''
        collapsers = self.modelspec_editor.collapsers
        controllers = self.modelspec_editor.controllers
        for col, con in zip(collapsers, controllers):
            con.hide()
        self.module_collapser.setArrowType(qc.Qt.RightArrow)

    def show_module_controls(self):
        '''Make module controls visible.'''
        collapsers = self.modelspec_editor.collapsers
        controllers = self.modelspec_editor.controllers
        for col, con in zip(collapsers, controllers):
            if not col.collapsed:
                con.show()
        self.module_collapser.setArrowType(qc.Qt.LeftArrow)

    def toggle_xfstep_controls(self):
        '''Hide xfstep controls if visible, or show them if hidden.'''
        if self.xfsteps_collapsed:
            self.show_xfstep_controls()
        else:
            self.hide_xfstep_controls()
        self.xfsteps_collapsed = not self.xfsteps_collapsed

    def hide_xfstep_controls(self):
        '''Make xfstep controls invisible.'''
        for s in self.xfspec_editor.steps:
            s.hide()
        self.xfstep_collapser.setArrowType(qc.Qt.LeftArrow)

    def show_xfstep_controls(self):
        '''Make xfstep controls visible.'''
        for s in self.xfspec_editor.steps:
            s.show()
        self.xfstep_collapser.setArrowType(qc.Qt.RightArrow)

    def toggle_bottom_controls(self):
        '''Hide Global and Fit controls if visible, or show them if hidden.'''
        if self.bottom_collapsed:
            self.fit_editor.show()
            self.global_controls.toggle_controls()
            self.bottom_collapser.setArrowType(qc.Qt.DownArrow)
        else:
            self.fit_editor.hide()
            self.global_controls.toggle_controls()
            self.bottom_collapser.setArrowType(qc.Qt.UpArrow)
        self.bottom_collapsed = not self.bottom_collapsed


class ModelspecEditor(qw.QWidget):
    def __init__(self, modelspec, rec, parent=None):
        '''
        QWidget for displaying per-module plots and editing model parameters.

        modelspec : ModelSpec
            A NEMS ModelSpec containing at least one module.
        rec : Recording
            A NEMS Recording, generally expected to contain 'stim', 'resp',
            and 'pred' signals.
        parent : QWidget*
            Expected to be an instance of EditorWidget.

        '''
        super(qw.QWidget, self).__init__()
        self.modelspec = modelspec
        self.original_modelspec = copy.deepcopy(modelspec)
        self.rec = rec
        self.parent = parent
        self.modules = []

    def setup_layout(self):
        '''
        Create a new layout with collapsers, controllers and plots

        Start a grid layout with an EpochsCollapser and EpochsWrapper
        on the first row.
        Then, for each module in self.modelspec, create a row with:
        [ModuleCollapser, ModuleController, ModuleCanvas],
        and add it to the grid layout.

        '''
        self.layout = qw.QGridLayout()
        self.modules = [ModuleCanvas(i, m, self)
                        for i, m in enumerate(self.modelspec.modules)]
        self.controllers = [ModuleControls(m, self) for m in self.modules]
        self.collapsers = [ModuleCollapser(m, self) for m in self.modules]

        widgets = zip(self.collapsers, self.controllers, self.modules)
        j = 0
        for col, cnt, m in widgets:
            if j == 0:
                self.epochs = EpochsWrapper(
                    recording=self.rec,
                    parent=self.parent.global_controls
                    )
                self.epochs_collapser = EpochsCollapser(self.epochs, self)
                self.layout.addWidget(self.epochs_collapser, 0, 0)
                self.layout.addWidget(self.epochs, 0, 2)
                j += 1

            self.layout.addWidget(col, j, 0)
            self.layout.addWidget(cnt, j, 1)
            self.layout.addWidget(m, j, 2)
            j += 1

        self.layout.setAlignment(qc.Qt.AlignTop)
        self.layout.setColumnStretch(2, 4)
        self.setLayout(self.layout)

    def refresh_plots(self):
        '''Regenerate plot for each module.'''
        for m in self.modules:
            m.new_plot()

    def evaluate_model(self):
        '''Evaluate modelspec and regenerate plots using updated recording.'''
        new_rec = self.parent.modelspec.evaluate()
        self.parent.modelspec.recording = new_rec
        self.modelspec.recording = new_rec
        self.refresh_plots()

    def reset_model(self):
        '''Reassign modelspec to original copy and regenerate layout.'''
        self.modelspec = copy.deepcopy(self.original_modelspec)
        self.clear_layout()
        self.setup_layout()
        self.refresh_plots()
        self.epochs.setup_figure()

    def clear_layout(self):
        temp = qw.QWidget()
        temp.setLayout(self.layout)


class ModuleCanvas(qw.QFrame):

    def __init__(self, mod_index, data, parent):
        '''
        QWidget for dispalying a module plot

        Parameters
        ----------
        mod_index : int
            Index of this canvas's module within parent editor's ModelSpec
        data : dict
            A single module from a NEMS ModelSpec.
        parent : QWidget*
            Expected to be an instance of ModelspecEditor.

        Attributes
        ----------
        plot_fn_idx : int
            Index for plot_fns list in data to use for choosing a plot type.
        fit_index : int
            fit_index from parent editor's modelspec.
        sig_name : str
            Name of signal to use for plotting.
        scrollable : boolean
            Determines whether or not plot can be scrolled in time by
            the GlobalControls slider.

        '''
        super(qw.QFrame, self).__init__()
        self.mod_index = mod_index
        self.parent = parent
        self.setFrameStyle(qw.QFrame.Panel | qw.QFrame.Sunken)

        # Default plot options - set them up here then change w/ controller
        self.plot_fn_idx = data.get('plot_fn_idx', 0)
        self.plot_channel = parent.modelspec.plot_channel
        self.fit_index = parent.modelspec.fit_index
        # TODO: Need to do something smarter for signal name
        self.sig_name = 'pred'
        self.scrollable = self.check_scrollable()

        self.layout = qw.QHBoxLayout()
        self.canvas = qw.QWidget()
        self.layout.addWidget(self.canvas)
        self.layout.setAlignment(qc.Qt.AlignTop)
        self.setLayout(self.layout)

    def new_plot(self):
        '''Remove plot from layout and replace it with a new one.'''
        self.layout.removeWidget(self.canvas)
        self.canvas.close()
        self.canvas = NemsCanvas(parent=self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.plot_on_axes()
        self.layout.addWidget(self.canvas)
        self.scrollable = self.check_scrollable()
        self.update_plot()

    def plot_on_axes(self):
        '''Draw plot on current canvas axes.'''
        ax = self.canvas.figure.add_subplot(111)
        rec = self.parent.modelspec.recording
        self.parent.modelspec.plot(self.mod_index, rec, ax,
                                   self.plot_fn_idx, self.fit_index,
                                   self.sig_name, no_legend=True,
                                   channels=self.plot_channel)
        self.canvas.draw()

    def check_scrollable(self):
        '''Set self.scrollable based on if current plot type is scrollable.'''
        plots = self.parent.modelspec[self.mod_index].get(
                'plot_fns', ['nems.plots.api.mod_output']
                )

        if plots[self.plot_fn_idx] in _SCROLLABLE_PLOT_FNS:
            scrollable = True
        else:
            scrollable = False
        return scrollable

    def update_plot(self):
        '''Shift xlimits of current plot if it's scrollable.'''
        if self.scrollable:
            gc = self.parent.parent.global_controls
            self.canvas.axes.set_xlim(gc.start_time, gc.stop_time)
            self.canvas.draw()
        else:
            pass

    def change_channel(self, value):
        self.plot_channel = int(value)
        self.new_plot()


class EpochsWrapper(qw.QFrame):

    def __init__(self, recording=None, parent=None):
        '''Adds a layout around an EpochCanvas to match ModuleCanvas.'''
        super(qw.QFrame, self).__init__()
        self.recording = recording
        self.epoch_parent = parent

    def setup_figure(self):
        '''Create a layout and add an EpochCanvas.'''
        self.layout = qw.QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.epochs = EpochCanvas(recording=self.recording,
                                  parent=self.epoch_parent)
        self.layout.addWidget(self.epochs)
        self.setLayout(self.layout)
        self.setFrameStyle(qw.QFrame.Panel | qw.QFrame.Sunken)

    def update_figure(self):
        '''Shifts xlim of epoch plot according to GlobalControls slider.'''
        self.epochs.update_figure()


class ModuleCollapser(qw.QWidget):

    def __init__(self, module, parent):
        '''Button for controlling visibility of associated module in editor.'''
        super(qw.QWidget, self).__init__()
        self.module = module
        self.controller = parent.controllers[module.mod_index]
        self.parent = parent
        self.collapsed = False

        layout = qw.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.toggle = qw.QPushButton('-', self)
        self.toggle.setFixedSize(12, 12)
        self.toggle.clicked.connect(self.toggle_collapsed)
        layout.addWidget(self.toggle)
        layout.setAlignment(qc.Qt.AlignTop)
        self.setLayout(layout)

    def toggle_collapsed(self):
        '''Toggle visibility of ModuleCanvas and ModuleController.'''
        if self.collapsed:
            self.module.show()
            if not self.parent.parent.modules_collapsed:
                self.controller.show()
            self.toggle.setText('-')
        else:
            self.module.hide()
            self.controller.hide()
            self.toggle.setText('+')
        self.collapsed = not self.collapsed


class EpochsCollapser(qw.QWidget):

    def __init__(self, epochs, parent):
        '''Button for controlling visibility of EpochsWrapper.'''
        super(qw.QWidget, self).__init__()
        self.epochs = epochs
        self.parent = parent
        self.collapsed = False

        layout = qw.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.toggle = qw.QPushButton('-', self)
        self.toggle.setFixedSize(12, 12)
        self.toggle.clicked.connect(self.toggle_collapsed)
        layout.addWidget(self.toggle)
        layout.setAlignment(qc.Qt.AlignTop)
        self.setLayout(layout)

    def toggle_collapsed(self):
        '''Toggle visibility of EpochsWrapper.'''
        if self.collapsed:
            self.epochs.show()
            self.toggle.setText('-')
        else:
            self.epochs.hide()
            self.toggle.setText('+')
        self.collapsed = not self.collapsed


class ModuleControls(qw.QFrame):

    def __init__(self, module, parent=None):
        '''QWidget for choosing module plot type and editing parameters.'''
        super(qw.QFrame, self).__init__()
        self.module = module
        self.parent = parent
        self.mod_index = self.module.mod_index
        self.module_data = copy.deepcopy(
                self.module.parent.modelspec[self.mod_index]
                )
        self.setFrameStyle(qw.QFrame.Panel | qw.QFrame.Raised)

        self.layout = qw.QVBoxLayout()

        name = self.module_data['fn']
        self.label = qw.QLabel(name)
        self.label.setFixedSize(330, 20)
        self.label.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.layout.addWidget(self.label)

        plot_layout = qw.QHBoxLayout()

        plot_list = self.module_data.get('plot_fns', [])
        self.plot_functions_menu = qw.QComboBox()
        self.plot_functions_menu.addItems(plot_list)
        initial_index = self.module.plot_fn_idx
        if initial_index is None:
            initial_index = 0
        self.plot_functions_menu.setCurrentIndex(initial_index)
        self.plot_functions_menu.setFixedSize(250, 24)

        plot_channel_layout = qw.QHBoxLayout()
        self.decrease_channel_btn = qw.QPushButton('-')
        self.decrease_channel_btn.clicked.connect(self.decrease_channel)
        self.decrease_channel_btn.setFixedSize(15, 15)
        self.channel_entry = qw.QLineEdit(str(self.module.plot_channel))
        self.channel_entry.textChanged.connect(self.change_channel)
        self.channel_entry.setFixedSize(30, 15)
        self.increase_channel_btn = qw.QPushButton('+')
        self.increase_channel_btn.clicked.connect(self.increase_channel)
        self.increase_channel_btn.setFixedSize(15, 15)
        plot_channel_layout.addWidget(self.decrease_channel_btn)
        plot_channel_layout.addWidget(self.channel_entry)
        plot_channel_layout.addWidget(self.increase_channel_btn)

        plot_layout.addWidget(self.plot_functions_menu)
        plot_layout.addLayout(plot_channel_layout)
        self.layout.addLayout(plot_layout)
        self.plot_functions_menu.currentIndexChanged.connect(self.change_plot)

        button_layout = qw.QHBoxLayout()
        self.edit_phi_btn = qw.QPushButton('Edit Phi')
        self.edit_phi_btn.clicked.connect(self.edit_phi)
        self.reset_phi_btn = qw.QPushButton('Reset Phi')
        self.reset_phi_btn.clicked.connect(self.reset_phi)
        button_layout.addWidget(self.edit_phi_btn)
        button_layout.addWidget(self.reset_phi_btn)
        self.layout.addLayout(button_layout)

        self.phi_editor = PhiEditor(self.module_data['phi'], self)
        self.phi_editor.hide()
        self.save_phi_btn = qw.QPushButton('Save Phi')
        self.save_phi_btn.hide()
        self.save_phi_btn.clicked.connect(self.save_phi)
        self.layout.addWidget(self.phi_editor)
        self.layout.addWidget(self.save_phi_btn)

        self.layout.setAlignment(qc.Qt.AlignTop)
        self.setLayout(self.layout)

    def change_plot(self, index):
        '''Change plot type according to plot_fn_idx and regenerate plot.'''
        self.module.plot_fn_idx = int(index)
        self.module.new_plot()

    def edit_phi(self):
        '''Show PhiEditor and 'Save Phi' button.'''
        self.phi_editor.show()
        self.save_phi_btn.show()

    def save_phi(self):
        '''Save phi values in PhiEditor and hide 'Save Phi' button.'''
        new_phi = self.phi_editor.export_phi()
        if not self.phi_equal(new_phi):
            need_evaluate = True
        else:
            need_evaluate = False

        self.parent.modelspec[self.mod_index]['phi'] = copy.deepcopy(new_phi)
        self.phi_editor.hide()
        self.save_phi_btn.hide()

        # Only re-evaluate if phi has changed from previous version.
        if need_evaluate:
            self.parent.evaluate_model()
            self.module_data['phi'] = copy.deepcopy(new_phi)

    def reset_phi(self):
        '''Set all phi for this module to original values.'''
        self.phi_editor.reset_phi()
        self.save_phi()

    def phi_equal(self, phi2):
        '''Test equality of module phi and phi2.'''
        equal = True
        phi1 = self.parent.modelspec[self.mod_index]['phi']
        for v1, v2 in zip(phi1.values(), phi2.values()):
            if not np.array_equal(v1, v2):
                equal = False
                break
        return equal

    def change_channel(self):
        old_channel = self.module.plot_channel
        new_channel = self.channel_entry.text()
        try:
            self.module.change_channel(int(new_channel))
        except:
            # Leaving this bare b/c not clear what exception type it will be.
            # But reason is if plot channel is not valid for whichever
            # plot function the module is using.
            log.warning("Invalid plot channel: %s for module: %s"
                        % (new_channel, self.mod_index))
            self.channel_entry.setText(str(old_channel))

    def decrease_channel(self):
        old_channel = self.module.plot_channel
        self.channel_entry.setText(str(max(0, old_channel-1)))

    def increase_channel(self):
        # TODO: Would be nice to know a valid channel range but
        #       I'm not sure how to extract that information from the modules.
        old_channel = self.module.plot_channel
        self.channel_entry.setText(str((old_channel+1)))


class PhiEditor(qw.QWidget):

    def __init__(self, phi, parent):
        '''QWidget for editing module parameters.'''
        super(qw.QWidget, self).__init__(parent)
        self.phi = phi
        self.original_phi = copy.deepcopy(phi)
        self.parent = parent

        self.setup_layout()

    def setup_layout(self):
        '''Add a layout with an ArrayModel for each phi entry.'''
        self.layout = qw.QFormLayout()
        self.arrays = {}
        for k, v in self.phi.items():
            label = qw.QLabel(k)
            array = ArrayModel(self, v)
            self.arrays[k] = array
            self.layout.addRow(label, array)

        self.layout.setAlignment(qc.Qt.AlignTop)
        self.setLayout(self.layout)

    def reset_phi(self):
        '''Reset all phi entries to their original values.'''
        self.phi = copy.deepcopy(self.original_phi)
        # remove old layout
        temp = qw.QWidget()
        temp.setLayout(self.layout)
        # recreate layout with new phi
        self.setup_layout()

    def export_phi(self):
        '''Return all phi entries and their current values as a dict.'''
        return {k: v.export_array() for k, v in self.arrays.items()}


class XfspecEditor(qw.QWidget):

    def __init__(self, xfspec, parent=None):
        '''(TODO)QWidget for editing xfspec options and toggling steps.'''
        super(qw.QWidget, self).__init__()
        self.xfspec = xfspec
        self.original_xfspec = copy.deepcopy(xfspec)
        self.parent = parent

        self.outer_layout = qw.QHBoxLayout()

        self.steps = [XfStepEditor(i, s, self)
                      for i, s in enumerate(self.xfspec)]
        self.step_layout = qw.QVBoxLayout()
        [self.step_layout.addWidget(s) for s in self.steps]

        self.outer_layout.addLayout(self.step_layout)
        self.setLayout(self.outer_layout)

    def filtered_xfspec(self):
        '''Return a nested list containing only xfsteps that are checked.'''
        checks = [s.checked for s in self.steps]
        x = [s for s, c in zip(self.xfspec, checks) if c]
        return x


class XfStepEditor(qw.QFrame):

    def __init__(self, index, step, parent):
        '''(TODO)QWidget for editing the parameters of one xforms step.'''
        super(qw.QFrame, self).__init__()
        self.index = index
        self.step = step
        self.parent = parent
        self.checked = True
        self.setFrameStyle(qw.QFrame.Panel | qw.QFrame.Raised)

        layout = qw.QVBoxLayout()
        self.checkbox = qw.QCheckBox(self.step[0], self)
        self.checkbox.setCheckState(qc.Qt.Checked)
        self.checkbox.stateChanged.connect(self.toggle)
        layout.addWidget(self.checkbox)
        self.setLayout(layout)

    def toggle(self):
        self.checked = not self.checked


class GlobalControls(qw.QFrame):

    start_time = 0
    display_duration = 10.0
    minimum_duration = 0.001
    stop_time = 10
    slider_scaling = 6
    max_time = 5000  # Larger value -> smoother scrolling

    def __init__(self, parent):
        '''QWidget for controlling cell_index, fit_index, and plot xlims.'''
        super(qw.QFrame, self).__init__()
        self.parent = parent
        self.collapsed = False
        self.setFrameStyle(qw.QFrame.Panel | qw.QFrame.Raised)

        # Slider for plot view windows
        self.time_slider = qw.QScrollBar(orientation=1)
        policy = qw.QSizePolicy()
        policy.setHorizontalPolicy(qw.QSizePolicy.Expanding)
        self.time_slider.setSizePolicy(policy)
        self._update_max_time()
        self.time_slider.setRange(0, self.max_time)
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
        self.range_layout = qw.QHBoxLayout()
        self.range_layout.setAlignment(qc.Qt.AlignTop)
        [self.range_layout.addWidget(w) for w in [self.display_range, plus, minus]]

        self.buttons_layout = qw.QHBoxLayout()
        self.buttons_layout.setAlignment(qc.Qt.AlignTop)
        self.reset_model_btn = qw.QPushButton('Reset Model')
        self.reset_model_btn.clicked.connect(self.reset_model)

        self.fit_index_label = qw.QLabel('Fit Index')
        self.fit_index_line = qw.QLineEdit()
        self.fit_index_line.editingFinished.connect(self.update_fit_index)
        self.fit_index_line.setText(str(self.parent.modelspec_editor.modelspec.fit_index))

        self.cell_index_label = qw.QLabel('Cell Index')
        self.cell_index_line = qw.QLineEdit()
        self.cell_index_line.editingFinished.connect(self.update_cell_index)
        self.cell_index_line.setText(str(self.parent.modelspec.plot_channel))
        #self.cell_index_line.setText(str(self.parent.modelspec_editor.modelspec.cell_index))

        self.buttons_layout.addWidget(self.reset_model_btn)
        self.buttons_layout.addWidget(self.fit_index_label)
        self.buttons_layout.addWidget(self.fit_index_line)
        self.buttons_layout.addWidget(self.cell_index_label)
        self.buttons_layout.addWidget(self.cell_index_line)

        layout = qw.QVBoxLayout()
        layout.setAlignment(qc.Qt.AlignTop)
        layout.addWidget(self.time_slider)
        layout.addLayout(self.range_layout)
        layout.addLayout(self.buttons_layout)
        self.setLayout(layout)

        #self._update_range()
        self.time_slider.setRange(0, self.max_time-self.display_duration)
        self.time_slider.setSingleStep(int(np.ceil(self.display_duration/10)))
        self.time_slider.setPageStep(int(self.display_duration))


    def scroll_all(self):
        '''Update xlims for all plots based on slider value.'''
        #self.start_time = self.time_slider.value()/self.slider_scaling
        self.start_time = self.time_slider.value()
        self.stop_time = self.start_time + self.display_duration

        # don't go past the latest time of the biggest plot
        # (should all have the same max most of the time)
        #self._update_max_time()
        #if self.stop_time >= self.max_signal_time:
        #    self.stop_time = self.max_signal_time
        #    self.start_time = max(0, self.max_signal_time - self.display_duration)
        #    #self.time_slider.setValue(0)

        [m.update_plot() for m in self.parent.modelspec_editor.modules]
        if len(self.parent.modelspec_editor.modules):
            self.parent.modelspec_editor.epochs.update_figure()

    def _update_max_time(self):
        resp = self.parent.rec.apply_mask()['resp']
        self.max_time = resp.as_continuous().shape[-1] / resp.fs
        self.max_signal_time = self.max_time
        self._update_range()
        #self.time_slider.setTickInterval(self.max_time/100)
        #self.time_slider.setRange(0, self.max_time-self.display_duration)
        #self.slider_scaling = self.max_time/(self.max_signal_time - self.display_duration)

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

    def reset_model(self):
        self.parent.modelspec_editor.reset_model()

    def update_fit_index(self):
        i = int(self.fit_index_line.text())
        j = self.parent.modelspec_editor.modelspec.fit_index

        if i == j:
            return

        if i > len(self.parent.modelspec_editor.modelspec.raw):
            # TODO: Flash red or something to indicate error
            self.fit_index_line.setText(str(j))
            return

        self.parent.modelspec_editor.modelspec.fit_index = i
        self.parent.modelspec_editor.evaluate_model()

    def update_cell_index(self):
        i = int(self.cell_index_line.text())
        j = self.parent.modelspec_editor.modelspec.cell_index

        if i == j:
           return

        if i > len(self.parent.modelspec_editor.modelspec.phis):
           # TODO: Flash red or something to indicate error
           self.cell_index_line.setText(str(j))
           return

        self.parent.modelspec_editor.modelspec.cell_index = i
        self.parent.modelspec_editor.evaluate_model()

    def toggle_controls(self):
        if self.collapsed:
            show_layout(self.buttons_layout)
            show_layout(self.range_layout)
        else:
            hide_layout(self.buttons_layout)
            hide_layout(self.range_layout)
        self.collapsed = not self.collapsed


class FitEditor(qw.QFrame):

    def __init__(self, parent):
        '''QWidget for manually initializing and fitting modelspec.'''
        super(qw.QFrame, self).__init__()
        self.parent = parent
        self.setFrameStyle(qw.QFrame.Panel | qw.QFrame.Raised)

        self.outer_layout = qw.QVBoxLayout()
        self.outer_layout.setAlignment(qc.Qt.AlignTop)

        self.init_layout = qw.QHBoxLayout()
        self.init_layout.setAlignment(qc.Qt.AlignTop)

        self.init_fn_menu = qw.QComboBox()
        self.init_fn_menu.addItems(_INIT_FNS)
        self.init_layout.addWidget(self.init_fn_menu)

        self.init_btn = qw.QPushButton('Initialize')
        self.init_btn.clicked.connect(self.initialize)
        self.init_layout.addWidget(self.init_btn)

        self.fit_layout = qw.QHBoxLayout()
        self.fit_layout.setAlignment(qc.Qt.AlignTop)

        self.fit_fn_menu = qw.QComboBox()
        self.fit_fn_menu.addItems(_FIT_FNS)
        self.fit_layout.addWidget(self.fit_fn_menu)

        self.fit_btn = qw.QPushButton('Fit')
        self.fit_btn.clicked.connect(self.fit)
        self.fit_layout.addWidget(self.fit_btn)

        self.iter_label = qw.QLabel('# iters')
        self.iter_edit = qw.QLineEdit('50')
        self.fit_layout.addWidget(self.iter_label)
        self.fit_layout.addWidget(self.iter_edit)

        self.outer_layout.addLayout(self.init_layout)
        self.outer_layout.addLayout(self.fit_layout)
        self.setLayout(self.outer_layout)

    def initialize(self):
        name = self.init_fn_menu.currentText()

        if 'from_keywords' in name:
            self.reset_from_keywords()
        else:
            fn = _lookup_fn_at(name)
            rec = self.parent.rec
            modelspec = self.parent.modelspec
            new_modelspec = fn(rec, modelspec)
            self.parent.set_new_modelspec(new_modelspec)

    def reset_from_keywords(self):
        fn = _lookup_fn_at('nems.initializers.from_keywords')
        s = self.parent.modelspec.modelspecname
        registry = self.parent.ctx.get('registry', None)
        rec = self.parent.rec
        meta = self.parent.modelspec.meta
        new_modelspec = fn(s, registry=registry, rec=rec, meta=meta)
        self.parent.set_new_modelspec(new_modelspec)

    def fit(self):
        name = self.fit_fn_menu.currentText()
        fn = _lookup_fn_at(name)
        n_iters = int(self.iter_edit.text())
        fit_kwargs = {'max_iter': n_iters}
        rec = self.parent.ctx['est']
        modelspec = self.parent.modelspec

        new_modelspec = fn(rec, modelspec, fit_kwargs=fit_kwargs)
        self.parent.set_new_modelspec(new_modelspec)


def hide_layout(layout):
    widgets = (layout.itemAt(i).widget() for i in
               range(layout.count()))
    for w in widgets:
        w.hide()

def show_layout(layout):
    widgets = (layout.itemAt(i).widget() for i in
               range(layout.count()))
    for w in widgets:
        w.show()


# Just for testing - typically will be opened by recording_browser.py
# ctx and xfspec should be loaded into current console environment elsewhere
def run(modelspec, xfspec, rec, ctx):
    app = qw.QApplication(sys.argv)
    ex = EditorWindow(modelspec=modelspec, xfspec=xfspec, rec=rec, ctx=ctx)
    sys.exit(app.exec_())

def browse_xform_fit(ctx, xfspec, recname='val'):

    modelspec=ctx['modelspec']
    rec=ctx[recname]
    ex = EditorWindow(modelspec=modelspec, xfspec=xfspec, rec=rec, ctx=ctx)

    return ex


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
        modelname = 'dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1'
        d = nd.get_results_file(batch, modelnames=[modelname], cellids=[cellid])
        filename = d.loc[0,'modelpath']
        xfspec, ctx = xforms.load_analysis(filename)
        modelspec = ctx['modelspec']
        rec = ctx['val']
    else:
        # If modelspec or rec aren't defined, set to None and let
        # EditorWindow try to set them based on ctx
        try:
            temp1 = modelspec
        except NameError:
            modelspec = None
        try:
            temp2 = rec
        except NameError:
            rec = None

    run(modelspec, xfspec, rec, ctx)
