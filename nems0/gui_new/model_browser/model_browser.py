import logging
import sys
from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from nems0.recording import Recording
from nems0.gui_new.model_browser.layer_area import LayerArea
from nems0.gui_new.model_browser.ui_promoted import CollapsibleBox, LeftDockWidget, PG_PLOTS

from nems0.modelspec import _lookup_fn_at

# TEMP ERROR CATCHER
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook


def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


# Set the exception hook to our wrapping function
sys.excepthook = my_exception_hook

log = logging.getLogger(__name__)

# read and parse the UI file to generate the GUI
qt_creator_file = Path(r'ui') / 'model_browser.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(qt_creator_file)


class MainWindow(QtBaseClass, Ui_MainWindow):

    def __init__(self, ctx=None, xfspec=None, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.setCentralWidget(None)

        self.ctx = ctx
        self.xfspec = xfspec

        self.rec_container = {}
        self.plot_container = {}
        self.cbox_container = []

        # add the layers menu
        self.menuView_addLayer = self.menuView.addMenu('Add layer')

        # add in menu items for the other signals
        self.menuView_addSignal = self.menuView.addMenu('Add signal')
        for k, v in self.ctx.items():
            if isinstance(v, Recording):
                rec_menu = self.menuView_addSignal.addMenu(k)
                rec_menu.triggered.connect(self.on_action_add_signal)
                # also add the rec to the rec container
                self.rec_container[k] = v
                for signal in v.signals:
                    add_signal = rec_menu.addAction(signal)
                    add_signal.setData(k)

        modelspec = self.ctx['modelspec']
        for idx, ms in enumerate(modelspec):
            module_name = ms['fn']

            # also add in the layer to the menu
            add_layer = self.menuView_addLayer.addAction(module_name)
            add_layer.triggered.connect(self.on_action_add_layer)

            layer_area = self.add_layer(idx)
            # expand the first layer
            if idx == 0:
                layer_area.parent().parent().set_toggle(True)

        # move the output to the end
        self.removeDockWidget(self.dockWidgetOutput)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dockWidgetOutput)
        self.dockWidgetOutput.show()

        # callbacks
        self.actionReset_panels.triggered.connect(self.on_action_reset_panels)
        self.actionSave_state.triggered.connect(self.save_state)

        self.init_plots()
        self.show()

        # save the state of all the docks so we can restore it
        self.save_state()

    def add_layer(self, idx):
        """Finds the layer in the modelspec and adds a dock for it."""
        ms = self.ctx['modelspec']
        layer_name = ms[idx]['fn']

        # get the layer output and keep track of it
        layer_output = ms.evaluate(self.ctx['val'], start=0, stop=idx + 1)
        self.rec_container[layer_name] = layer_output

        # make the layer area and add it as a dock
        layer_area = LayerArea(rec_name=layer_name, signal_names=['pred'])
        self.plot_container[id(layer_area)] = layer_area  # TODO: fix these keys
        self.add_collapsible_dock(layer_area, window_title=layer_name)

        # add the plot types to the combo box
        layer_area.comboBox.blockSignals(True)
        layer_area.comboBox.addItems(PG_PLOTS.keys())
        layer_area.comboBox.addItems(ms[idx]['plot_fns'])
        layer_area.comboBox.blockSignals(False)

        return layer_area

    def add_collapsible_dock(self, widget, window_title='', dock_title=None):
        """Adds a collapsible dock."""
        # create a dock to contain the collapsing box and layer area
        dock = LeftDockWidget(self, window_title=window_title, dock_title=dock_title)

        # make the collapsible area
        cbox = CollapsibleBox(parent=dock)
        self.cbox_container.append(cbox)
        layout = QVBoxLayout(dock)  # dock's layout for the cbox
        layout.setContentsMargins(0, 0, 0, 0)  # no margins so it can completely collapse

        # add the widget to the cbox and setup the widget's parent
        layout.addWidget(widget)
        widget.setParent(dock)
        # put it all together
        cbox.setContentLayout(layout)
        dock.setWidget(cbox)
        dock.connect_min(cbox.on_pressed)  # connect the min button in the dock bar to the cbox collapse
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def init_plots(self):
        """Populates the various plots with data."""
        self.link_plots()

        self.outputPlot.update_plot(rec_name='val', signal_names=['pred', 'resp'], channels=0)

        for layer_area in self.plot_container.values():
            layer_area.update_plot()

        self.inputSpectrogram.plot_input(rec_name='val')

    def link_plots(self):
        """Links the x region of plots."""
        self.link_together(self.outputPlot)

        for layer_area in self.plot_container.values():
            self.link_together(layer_area.plotWidget)

    def link_together(self, output_plot, finished=False):
        """Links together the input plot to an output plot."""
        self.inputSpectrogram.add_link(output_plot.updateXRange, finished=finished)

        self.inputSpectrogram.lr.sigRegionChanged.emit(self.inputSpectrogram.lr)
        self.inputSpectrogram.lr.sigRegionChangeFinished.emit(self.inputSpectrogram.lr)

        output_plot.add_link(self.inputSpectrogram.updateRegion)

    def unlink(self, output_plot):
        """Unlinks plots."""
        self.inputSpectrogram.unlink(output_plot.updateXRange)
        output_plot.unlink(self.inputSpectrogram.updateRegion)

    def save_state(self):
        self.saved_state = self.saveState(0)
        # self.saved_geometry = self.saveGeometry()

        # record the hidden/toggle status of collapsible docks
        self.toggle_status = [(cbox.isVisible(), cbox.is_open) for cbox in self.cbox_container]

    def on_action_reset_panels(self):
        """Relays out the panels as they were on program startup."""
        # TODO: get the order to be respected (need to change the actual order in init instead of just remove/add the
        #  output)
        self.restoreState(self.saved_state, )
        # self.restoreGeometry(self.saved_geometry)

        for cbox, (visible, toggle) in zip(self.cbox_container, self.toggle_status):
            cbox.set_toggle(toggle)
            if visible:
                cbox.parent().show()

    def on_action_add_layer(self):
        layers = [ms['fn'] for ms in self.ctx['modelspec']]
        layer_name = self.sender().text()

        layer_area = self.add_layer(layers.index(layer_name))
        layer_area.update_plot()
        self.link_together(layer_area.plotWidget)
        layer_area.parent().parent().set_toggle(True)

    def on_action_add_signal(self, action):
        signal, recording = action.text(), action.data()

        # make the layer area and add it to the dock
        layer_area = LayerArea(rec_name=recording, signal_names=[signal])
        self.plot_container[id(layer_area)] = layer_area
        self.add_collapsible_dock(layer_area, window_title=f'{recording}:{signal}')

        # layer_area.plotWidget.update_plot(y_data=signal_data, y_data_name=signal)
        layer_area.parent().parent().set_toggle(True)

        # add the plot types to the combo box
        layer_area.comboBox.blockSignals(True)
        layer_area.comboBox.addItems(PG_PLOTS.keys())
        layer_area.comboBox.blockSignals(False)

        # only link if shapes match
        ## TODO: make this a try in the plot widget
        # if self.ctx['val']['stim']._data.shape[-1] == signal_data.shape[-1]:
        #     self.link_together(layer_area.plotWidget)

        layer_area.update_plot()


if __name__ == '__main__':
    from nems0 import xforms
    demo_model = '/auto/data/nems_db/results/322/ARM030a-28-2/ozgf.fs100.ch18-ld-sev.dlog-wc.18x3.g-fir.3x15-lvl.1-dexp.1.tfinit.n.lr1e3.rb5.es20-newtf.n.lr1e4.es20.2021-06-15T212246'
    xfspec, ctx = xforms.load_analysis(demo_model)

    #xfspec, ctx = xforms.load_context(r'C:\Users\Alex\PycharmProjects\NEMS\results\temp_xform')

    app = QApplication(sys.argv)
    window = MainWindow(ctx=ctx, xfspec=xfspec)
    app.exec_()
