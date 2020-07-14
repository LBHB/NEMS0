import logging
import sys
from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from nems.gui_new.model_browser.layer_area import LayerArea
from nems.gui_new.model_browser.ui_promoted import CollapsibleBox

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

        self.ctx = ctx
        self.xfspec = xfspec

        # add layout and collapsible boxes
        content = QWidget(parent=self.scrollArea)
        self.scrollArea.setWidget(content)
        self.scrollArea.setWidgetResizable(True)

        vlayout = QVBoxLayout(content)

        modules = [ms['fn'] for ms in self.ctx['modelspec']]
        for module in modules:
            cbox = CollapsibleBox(f'{module}', parent=self.scrollArea)
            vlayout.addWidget(cbox)

            box_layout = QVBoxLayout(self)
            layer_area = LayerArea(parent=self.scrollArea, name='testing')
            box_layout.addWidget(layer_area)

            cbox.setContentLayout(box_layout)

        vlayout.addStretch()

    def init_plots(self):
        """Populates the various plots with data"""
        self.link_plots()

        output_pred = ctx['val']['pred']._data
        output_resp = ctx['val']['resp']._data
        self.outputPlot.update_plot(y_data=output_pred, y_data2=output_resp)

        self.inputSpectrogram.plot_input(ctx['val'])

    def link_plots(self):
        """Links the x region of plots."""
        self.outputPlot.set_xlink(self.inputSpectrogram.p2)


if __name__ == '__main__':
    from nems import xforms
    analysis_path = Path(
        r'C:\Users\Alex\PycharmProjects\NEMS\results\308\BRT034f-07-2\ozgf.fs100.ch18-ld-sev.dlog-wc.18x2.g-fir.1x15x2-relu.2-wc.2x1.z-lvl.1-dexp.1.newtf.n.i.lr5e3.et6.2020-06-17T213522')
    xfspec, ctx = xforms.load_analysis(str(analysis_path), eval_model=True)
    # xfspec, ctx = None, None

    app = QApplication(sys.argv)
    window = MainWindow(ctx=ctx, xfspec=xfspec)
    window.show()
    window.init_plots()
    app.exec_()
