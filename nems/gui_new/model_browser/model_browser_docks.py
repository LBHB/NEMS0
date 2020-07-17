import logging
import sys
from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

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

        self.rec_container = {}
        self.plot_container = {}
        self.cbox_container = []

        modelspec = self.ctx['modelspec']
        for idx, ms in enumerate(modelspec):
            module_name = ms['fn']

            # get the layer output
            layer_output = modelspec.evaluate(self.ctx['val'], start=0, stop=idx + 1)
            self.rec_container[module_name] = layer_output['pred']._data

            # create a dock to contain the collapsing box and layer area
            dock = LeftDockWidget(self, title=module_name)

            # make the collapsible area with the contained area
            cbox = CollapsibleBox(parent=dock)
            self.cbox_container.append(cbox)
            layout = QVBoxLayout(dock)
            layout.setContentsMargins(0, 0, 0, 0)
            layer_area = LayerArea(parent=dock, name='another test')
            self.plot_container[module_name] = layer_area.plotWidget

            # put it together
            layout.addWidget(layer_area)
            cbox.setContentLayout(layout)
            dock.setWidget(cbox)
            dock.connect_min(cbox.on_pressed)
            self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        # move the output to the end
        self.removeDockWidget(self.dockWidgetOutput)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dockWidgetOutput)
        self.dockWidgetOutput.show()

        # callbacks
        self.actionReset_panels.triggered.connect(self.on_action_reset_panels)

        self.show()
        self.init_plots()

        # save the state of all the docks so we can restore it
        self.save_state()

    def init_plots(self):
        """Populates the various plots with data"""
        self.link_plots()

        output_pred = self.ctx['val']['pred']._data
        output_resp = self.ctx['val']['resp']._data
        self.outputPlot.update_plot(y_data=output_pred, y_data2=output_resp)

        for module_name, plot in self.plot_container.items():
            plot.update_plot(y_data=self.rec_container[module_name])

        self.inputSpectrogram.plot_input(self.ctx['val'])

    def link_plots(self):
        """Links the x region of plots."""
        self.link_together(self.outputPlot)

        for output_plot in self.plot_container.values():
            self.link_together(output_plot)

    def link_together(self, output_plot):
        """Links together the input plot to an output plot."""
        self.inputSpectrogram.add_link(output_plot.updateXRange)
        output_plot.add_link(self.inputSpectrogram.updateRegion)

    def save_state(self):
        self.saved_state = self.saveState(1)
        # self.saved_geometry = self.saveGeometry()

        # record the toggle status of collapsible docks
        self.toggle_status = [cbox.is_open for cbox in self.cbox_container]

    def on_action_reset_panels(self):
        """Relays out the panels as they were on program startup."""
        self.restoreState(self.saved_state, 1)
        # self.restoreGeometry(self.saved_geometry)

        for cbox, toggle in zip(self.cbox_container, self.toggle_status):
            cbox.set_toggle(toggle)
            cbox.parent().show()


if __name__ == '__main__':
    from nems import xforms
    analysis_path = Path(
        r'C:\Users\Alex\PycharmProjects\NEMS\results\308\BRT034f-07-2\ozgf.fs100.ch18-ld-sev.dlog-wc.18x2.g-fir.1x15x2-relu.2-wc.2x1.z-lvl.1-dexp.1.newtf.n.i.lr5e3.et6.2020-06-17T213522')
    xfspec, ctx = xforms.load_analysis(str(analysis_path), eval_model=True)
    # xfspec, ctx = None, None

    app = QApplication(sys.argv)
    window = MainWindow(ctx=ctx, xfspec=xfspec)
    # window.show()
    # window.init_plots()
    app.exec_()