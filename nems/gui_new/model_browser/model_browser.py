import logging
import sys
from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from nems.recording import Recording
from nems.gui_new.model_browser.layer_area import LayerArea
from nems.gui_new.model_browser.ui_promoted import CollapsibleBox, LeftDockWidget

from nems.modelspec import _lookup_fn_at

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
qt_creator_file = Path(r'ui') / 'model_browser_docks.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(qt_creator_file)


class MainWindow(QtBaseClass, Ui_MainWindow):

    def __init__(self, ctx=None, xfspec=None, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.setCentralWidget(None)

        self.ctx = ctx
        self.xfspec = xfspec

        self.signal_container = {}
        self.plot_container = {}
        self.cbox_container = []

        self.menuView_addLayer = self.menuView.addMenu('Add layer')

        modelspec = self.ctx['modelspec']
        for idx, ms in enumerate(modelspec):
            module_name = ms['fn']

            # also add in the layer to the menu
            add_layer = self.menuView_addLayer.addAction(module_name)
            add_layer.triggered.connect(self.on_action_add_layer)

            self.add_layer(idx)

        # add in menu items for the other signals
        self.menuView_addSignal = self.menuView.addMenu('Add signal')
        for k, v in self.ctx.items():
            if isinstance(v, Recording):
                rec_menu = self.menuView_addSignal.addMenu(k)
                rec_menu.triggered.connect(self.on_action_add_signal)
                for signal in v.signals:
                    add_signal = rec_menu.addAction(signal)
                    add_signal.setData(k)

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
        self.signal_container[layer_name] = layer_output['pred']._data

        # make the layer area and add it as a dock
        layer_area = LayerArea(layer_name=layer_name)
        self.plot_container[id(layer_area)] = layer_area  # TODO: fix these keys
        self.add_collapsible_dock(layer_area, window_title=layer_name)

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

        output_pred = self.ctx['val']['pred']._data
        output_resp = self.ctx['val']['resp']._data
        self.outputPlot.update_plot(y_data=output_pred, y_data2=output_resp,
                                    y_data_name='pred', y_data2_name='resp')

        for layer_area in self.plot_container.values():
            layer_area.plotWidget.update_plot(y_data=self.signal_container[layer_area.layer_name],
                                              y_data_name='pred')

        self.inputSpectrogram.plot_input(self.ctx['val'])

    def link_plots(self):
        """Links the x region of plots."""
        self.link_together(self.outputPlot)

        for layer_area in self.plot_container.values():
            self.link_together(layer_area.plotWidget)

    def link_together(self, output_plot):
        """Links together the input plot to an output plot."""
        self.inputSpectrogram.add_link(output_plot.updateXRange)
        self.inputSpectrogram.lr.sigRegionChanged.emit(self.inputSpectrogram.lr)
        output_plot.add_link(self.inputSpectrogram.updateRegion)

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
        layer_area.plotWidget.update_plot(y_data=self.signal_container[layer_name], y_data_name='pred')
        self.link_together(layer_area.plotWidget)
        layer_area.parent().parent().set_toggle(True)

    def on_action_add_signal(self, action):
        signal, recording = action.text(), action.data()
        signal_data = self.ctx[recording][signal].rasterize()._data
        signal_name = f'{recording}:{signal}'
        self.signal_container[signal_name] = signal_data

        # make the layer area and add it to the dock
        layer_area = LayerArea(layer_name=signal_name)
        self.plot_container[id(layer_area)] = layer_area
        self.add_collapsible_dock(layer_area, window_title=signal_name)

        layer_area.plotWidget.update_plot(y_data=signal_data, y_data_name=signal)
        layer_area.parent().parent().set_toggle(True)

        # only link if shapes match
        if self.ctx['val']['stim']._data.shape[-1] == signal_data.shape[-1]:
            self.link_together(layer_area.plotWidget)


if __name__ == '__main__':
    from nems import xforms
    xfspec, ctx = xforms.load_context(r'C:\Users\Alex\PycharmProjects\NEMS\results\temp_xform')

    app = QApplication(sys.argv)
    window = MainWindow(ctx=ctx, xfspec=xfspec)
    app.exec_()
