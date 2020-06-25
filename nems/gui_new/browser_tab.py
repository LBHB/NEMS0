import logging
import sys
from configparser import ConfigParser, DuplicateSectionError
from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from nems import get_setting, xforms
from nems.gui import editors
from nems.gui_new.ui_promoted import ListViewListModel

log = logging.getLogger(__name__)

# read and parse the UI file to generate the GUI
qt_creator_file = Path(r'ui') / 'tab_browser.ui'
Ui_Widget, QtBaseClass = uic.loadUiType(qt_creator_file)


class BrowserTab(QtBaseClass, Ui_Widget):

    def __init__(self, parent=None):
        super(BrowserTab, self).__init__(parent)
        self.setupUi(self)

        self.tab_name = 'browser_tab'

    def init_models(self):
        """Initialize the models."""
        # setup the data and models
        self.batchModel = self.parent.batchModel
        self.comboBoxBatches.setModel(self.batchModel)

        # current_batch = self.batchModel.data[self.comboBoxBatches.currentIndex()][0]
        current_batch = self.batchModel.index(self.comboBoxBatches.currentIndex()).data()
        cell_filter = {'batch': current_batch}
        self.cellsModel = ListViewListModel(table_name='Results', column_name='cellid', filter=cell_filter)
        self.comboBoxCells.setModel(self.cellsModel)

        # current_cell = self.cellsModel.data[self.comboBoxCells.currentIndex()][0]
        current_cell = self.cellsModel.index(self.comboBoxCells.currentIndex()).data()
        model_filter = {'batch': current_batch, 'cellid': current_cell}
        self.modelnameModel = ListViewListModel(table_name='Results', column_name='modelname', filter=model_filter)
        self.listViewModelnames.setModel(self.modelnameModel)
        # select first entry
        self.listViewModelnames.selectionModel().setCurrentIndex(self.modelnameModel.index(0),
                                                                 QItemSelectionModel.SelectCurrent)

        # keep some references
        self.batch = current_batch
        self.cellid = current_cell
        self.modelname = self.modelnameModel.index(0).data()

        # keep track of the all the models with db connections
        self.parent.db_conns.extend([self.cellsModel, self.modelnameModel])

        # setup the callbacks for viewers
        self.comboBoxBatches.currentIndexChanged.connect(self.on_batch_changed)
        self.comboBoxCells.currentIndexChanged.connect(self.on_cell_changed)
        self.listViewModelnames.selectionModel().currentChanged.connect(self.on_modelname_changed)

        # setup the callbacks for the buttons
        self.pushButtonViewModel.clicked.connect(self.on_view_model)

        # make some parent stuff available locally for ease
        self.statusbar = self.parent.statusbar
        self.config = self.parent.config

        # update inputs
        self.update_selections(*self.load_settings())

    def update_selections(self, batch=None, cellid=None, modelname=None):
        """Sets the inputs of batch, cellid, and modelname appropriately.

        We want to avoid triggering events for all of these updates, so go through and disconnect and
        don't update the selection cascade to avoid triggering those updates until we can disable those callbacks."""
        if batch is not None:
            batch_index = self.comboBoxBatches.findText(str(batch), Qt.MatchFixedString)
            if batch_index >= 0:
                self.batch = batch
                self.comboBoxBatches.disconnect()
                self.comboBoxBatches.setCurrentIndex(batch_index)
                self.on_batch_changed(batch_index, update_selection=True if cellid is None else False)
                self.comboBoxBatches.currentIndexChanged.connect(self.on_batch_changed)

        if cellid is not None:
            cellid_index = self.comboBoxCells.findText(cellid, Qt.MatchFixedString)
            if cellid_index >= 0:
                self.cellid = cellid
                self.comboBoxCells.disconnect()
                self.comboBoxCells.setCurrentIndex(cellid_index)
                self.on_cell_changed(cellid_index, update_selection=True if modelname is None else False)
                self.comboBoxCells.currentIndexChanged.connect(self.on_cell_changed)

        if modelname is not None:
            modelname_index = self.modelnameModel.get_index(modelname)
            if modelname_index >= 0:
                self.modelname = modelname
                model_index = self.modelnameModel.index(modelname_index)
                self.listViewModelnames.selectionModel().setCurrentIndex(model_index, QItemSelectionModel.SelectCurrent)
                self.listViewModelnames.scrollTo(model_index)

    def load_settings(self, group_name=None):
        """Get the tabs saved selections of batch, cellid, modelname."""
        config_group = self.parent.config_group
        if group_name is not None:
            config_group = config_group + ':' + group_name

        if config_group not in self.config:
            return None, None, None

        batch = self.config[config_group].get(f'{self.tab_name}:batch', None)
        batch = batch if batch is None else int(batch)
        cellid = self.config[config_group].get(f'{self.tab_name}:cellid', None)
        modelname = self.config[config_group].get(f'{self.tab_name}:modelname', None)

        return batch, cellid, modelname

    def get_selections(self):
        """Passes the tabs selections up to the parent for saving."""

        return {
            f'{self.tab_name}:batch': str(self.batch),
            f'{self.tab_name}:cellid': self.cellid,
            f'{self.tab_name}:modelname': self.modelname,
        }

    def on_action_open(self):
        """Event handler for action open."""
        directory = QFileDialog.getExistingDirectory(self, 'Load modelspec')
        if not directory:
            return

        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f'Could not find directory: "{directory}".')
        if not directory.is_dir():
            raise ValueError('Path should be a directory to a saved modelspec, not a file.')

        log.info(f'Loading modelspec from path: "{directory}".')
        self.statusbar.showMessage('Loading model...')  # don't timeout since we clear below

        xfspec, ctx = self.load_xfrom_from_folder(directory, eval_model=True)
        self.statusbar.clearMessage()
        self.launch_model_browser(ctx, xfspec)

    def on_batch_changed(self, index, update_selection=True):
        """Event handler for batch selection.

        Starts cascade of updating the proper UI elements. Looks to see if current cell exists
        in new batch and if so sets it, otherwise resets index to 0.
        """
        self.batch = self.batchModel.index(index).data()
        log.info(f'Batch changed to "{self.batch}", loading cellids.')

        old_cellid = self.comboBoxCells.currentText()
        filter = {'batch': self.batch}
        self.cellsModel.refresh_data(filter)

        # need to ensure the current index is in bounds before updating the combobox
        # but temporarily don't emit a signal since we'll send it below again
        self.comboBoxCells.disconnect()
        self.comboBoxCells.setCurrentIndex(-1)
        self.comboBoxCells.currentIndexChanged.connect(self.on_cell_changed)

        self.cellsModel.layoutChanged.emit()

        # look for old cell in new list of cells
        if update_selection:
            index = self.comboBoxCells.findText(old_cellid, Qt.MatchFixedString)
            if index == -1:
                index = 0
            self.comboBoxCells.setCurrentIndex(index)

        # do a manual redraw since slow to update sometimes, not sure why
        self.comboBoxCells.update()

    def on_cell_changed(self, index, update_selection=True):
        """Event handler for cellid change."""
        # self.cellid = self.cellsModel.data[index][0]
        self.cellid = self.cellsModel.index(index).data()
        log.info(f'Cellid changed to "{self.cellid}", loading modelnames.')

        # qcombobox and qlistview work a bit different in terms of accessing data
        old_modelname = self.listViewModelnames.selectedIndexes()[0].data()
        filter = {'batch': self.batch, 'cellid': self.cellid}
        self.modelnameModel.refresh_data(filter)

        # for qlistview, no issues with out of bounds, but we still want to disconnect and
        # reconnect so that we can ensure we emit a signal even if the index stays the same
        self.listViewModelnames.selectionModel().disconnect()
        model_index = self.modelnameModel.index(-1)
        self.listViewModelnames.selectionModel().setCurrentIndex(model_index,
                                                                 QItemSelectionModel.SelectCurrent)
        self.listViewModelnames.scrollTo(model_index)
        self.listViewModelnames.selectionModel().currentChanged.connect(self.on_modelname_changed)

        self.modelnameModel.layoutChanged.emit()

        # look for old modelname in new list of modelnames
        if update_selection:
            index = self.modelnameModel.get_index(old_modelname)
            if index == -1:
                index = 0
            model_index = self.modelnameModel.index(index)
            self.listViewModelnames.selectionModel().setCurrentIndex(model_index,
                                                                     QItemSelectionModel.SelectCurrent)
            self.listViewModelnames.scrollTo(model_index)

    def on_modelname_changed(self, current, previous):
        """Event handler for model change."""
        self.modelname = current.data()
        self.listViewModelnames.update()
        log.info(f'Modelname changed to "{self.modelname}".')

    def on_view_model(self):
        """Event handler for view model button."""
        self.statusbar.showMessage('Loading model...', 5000)

    def load_xfrom_from_folder(self, directory, eval_model=True):
        """Loads an xform/context from a directory."""
        xfspec, ctx = xforms.load_analysis(str(directory), eval_model=eval_model)
        return xfspec, ctx

    def launch_model_browser(self, ctx, xfspec):
        """Launches the model browser and keeps a reference so it's not garbage collected."""
        self.model_browser = editors.browse_xform_fit(ctx, xfspec)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BrowserTab()
    window .show()
    app.exec_()
