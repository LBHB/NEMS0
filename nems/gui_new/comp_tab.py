from pathlib import Path
import logging
import sys
from configparser import ConfigParser, DuplicateSectionError

from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from nems import xforms, get_setting
from nems.gui_new.ui_promoted import ListViewListModel

log = logging.getLogger(__name__)

# read and parse the UI file to generate the GUI
qt_creator_file = Path(r'ui') / 'tab_comp.ui'
Ui_Widget, QtBaseClass = uic.loadUiType(qt_creator_file)


class CompTab(QtBaseClass, Ui_Widget):

    def __init__(self, parent=None):
        super(CompTab, self).__init__(parent)
        self.setupUi(self)

        self.tab_name = 'comp_tab'

    def init_models(self):
        """Initialize the models."""
        # setup the data and models
        # self.batchModel = ListViewListModel(table_name='Results', column_name='batch')
        self.batchModel = self.parent.batchModel
        self.comboBoxBatches.setModel(self.batchModel)

        # current_batch = self.batchModel.data[self.comboBoxBatches.currentIndex()][0]
        current_batch = self.batchModel.index(self.comboBoxBatches.currentIndex()).data()
        batch_filter = {'batch': current_batch}
        self.modelnameModel = ListViewListModel(table_name='Results', column_name='modelname', filter=batch_filter)
        self.comboBoxModel1.setModel(self.modelnameModel)
        self.comboBoxModel2.setModel(self.modelnameModel)
        # set the second model to be the second value (so they're not the same)
        self.comboBoxModel2.setCurrentIndex(1)

        # keep some references
        self.batch = current_batch
        self.modelname1 = self.modelnameModel.index(self.comboBoxModel1.currentIndex()).data()
        self.modelname2 = self.modelnameModel.index(self.comboBoxModel2.currentIndex()).data()

        self.cellsModel = ListViewListModel(table_name='Results', column_name='cellid', filter=batch_filter)
        self.listViewCells.setModel(self.cellsModel)
        # select first entry
        self.listViewCells.selectionModel().setCurrentIndex(self.cellsModel.index(0),
                                                                 QItemSelectionModel.SelectCurrent)
        self.cellids = [self.cellsModel.index(i.row()).data() for i in self.listViewCells.selectedIndexes()]

        # keep track of the all the models with db connections
        self.parent.db_conns.extend([self.modelnameModel, self.cellsModel])

        # setup the callbacks for the viewers
        self.comboBoxBatches.currentIndexChanged.connect(self.on_batch_changed)
        self.comboBoxModel1.currentIndexChanged.connect(self.on_modelname_changed)
        self.comboBoxModel2.currentIndexChanged.connect(self.on_modelname_changed)
        self.listViewCells.selectionModel().selectionChanged.connect(self.on_cells_changed)

        # make some parent stuff available locally for ease
        self.statusbar = self.parent.statusbar
        self.config = self.parent.config

    def get_selections(self):
        return {}

    def update_selections(self):
        pass

    def load_settings(self, group_name=None):
        return {}

    def on_batch_changed(self, index, update_selection=True):
        """Event handler for batch selection.

        Starts cascade of updating the proper UI elements. Looks to see if current modelnames exists
        in new batch and if so sets them, otherwise resets index to 0 and 1.
        """
        self.batch = self.batchModel.index(index).data()
        log.info(f'Batch changed to "{self.batch}", loading modelnames.')

        old_modelname1 = self.comboBoxModel1.currentText()
        old_modelname2 = self.comboBoxModel2.currentText()
        filter = {'batch': self.batch}
        self.modelnameModel.refresh_data(filter)

        # need to ensure the current index is in bounds before updating the combobox
        # but temporarily don't emit a signal since we'll send it below again
        self.comboBoxModel1.disconnect()
        self.comboBoxModel2.disconnect()
        self.comboBoxModel1.setCurrentIndex(-1)
        self.comboBoxModel2.setCurrentIndex(-1)
        # however, only reconnect one of them now, and the other after the later signal so that we don't trigger
        # two events
        self.comboBoxModel1.currentIndexChanged.connect(self.on_modelname_changed)

        self.cellsModel.layoutChanged.emit()

        # look for old cell in new list of cells
        if update_selection:
            index1 = self.comboBoxModel1.findText(old_modelname1, Qt.MatchFixedString)
            index2 = self.comboBoxModel2.findText(old_modelname2, Qt.MatchFixedString)
            if index1 == -1:
                index1 = 0
            if index2 == -1:
                index2 = 1
            self.comboBoxModel1.setCurrentIndex(index1)
            self.comboBoxModel2.setCurrentIndex(index2)

        # do a manual redraw since slow to update sometimes, not sure why
        self.comboBoxModel1.update()
        self.comboBoxModel2.update()
        self.comboBoxModel2.currentIndexChanged.connect(self.on_modelname_changed)

    def on_modelname_changed(self, index, update_selection=True):
        """Event handler for model change."""
        self.modelname1 = self.modelnameModel.index(self.comboBoxModel1.currentIndex()).data()
        self.modelname2 = self.modelnameModel.index(self.comboBoxModel2.currentIndex()).data()
        log.info('Modelnames changed, loading cellids.')

        # here we don't keep track of the old cellids because since it's extended selection, it would be expensive
        # to try to search for all the matching cellids
        filter = {'batch': self.batch}
        self.cellsModel.refresh_data(filter)

        # for qlistview, no issues with out of bounds, but we still want to disconnect and
        # reconnect so that we can ensure we emit a signal even if the index stays the same
        self.listViewCells.selectionModel().disconnect()
        model_index = self.cellsModel.index(0)
        self.listViewCells.selectionModel().setCurrentIndex(model_index, QItemSelectionModel.SelectCurrent)
        self.listViewCells.scrollTo(model_index)
        self.listViewCells.selectionModel().selectionChanged.connect(self.on_cells_changed)

        self.cellsModel.layoutChanged.emit()

    def on_cells_changed(self, *args, **kwargs):
        """Event handler for model change."""
        self.cellids = [self.cellsModel.index(i.row()).data() for i in self.listViewCells.selectedIndexes()]
        print(self.cellids)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CompTab()
    window .show()
    app.exec_()
