import logging
import sys
from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from nems0 import xform_helper, xforms
from nems.gui import editors
from nems.gui_new.db_browser.ui_promoted import ListViewListModel
from nems.utils import lookup_fn_at

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
        # make some parent stuff available locally for ease
        self.statusbar = self.parent.statusbar
        self.config = self.parent.config

        # see if there's any config to load
        batch, cellid, modelname, custom_fn = self.load_settings()

        # setup the data and models
        self.batchModel = self.parent.batchModel
        self.comboBoxBatches.setModel(self.batchModel)
        # if batch passed in, set it here
        if batch is not None:
            batch_index = self.comboBoxBatches.findText(str(batch), Qt.MatchFixedString)
            self.comboBoxBatches.setCurrentIndex(batch_index)

        current_batch = self.batchModel.index(self.comboBoxBatches.currentIndex()).data()
        cell_filter = {'batch': current_batch}
        self.cellsModel = ListViewListModel(table_name='Results', column_name='cellid', filter=cell_filter)
        self.comboBoxCells.setModel(self.cellsModel)
        if cellid is not None:
            cellid_index = self.comboBoxCells.findText(cellid, Qt.MatchFixedString)
            self.comboBoxCells.setCurrentIndex(cellid_index)

        current_cell = self.cellsModel.index(self.comboBoxCells.currentIndex()).data()
        model_filter = {'batch': current_batch, 'cellid': current_cell}
        self.modelnameModel = ListViewListModel(table_name='Results', column_name='modelname', filter=model_filter)

        # setup the proxy filty model
        self.modelnamesProxyModel = QSortFilterProxyModel(self)
        self.modelnamesProxyModel.setSourceModel(self.modelnameModel)
        # self.modelnamesProxyModel.setFilterKeyColumn(0)
        self.modelnamesProxyModel.setDynamicSortFilter(True)
        self.modelnamesProxyModel.setFilterCaseSensitivity(Qt.CaseInsensitive)

        self.listViewModelnames.setModel(self.modelnamesProxyModel)
        if modelname is not None:
            modelname_index = self.modelnameModel.index(self.modelnameModel.get_index(modelname))
            if modelname_index.row() >= 0:
                self.listViewModelnames.selectionModel().setCurrentIndex(modelname_index,
                                                                         QItemSelectionModel.SelectCurrent)
                self.listViewModelnames.scrollTo(modelname_index)
                self.modelname = modelname
        else:
            # select first entry
            self.listViewModelnames.selectionModel().setCurrentIndex(self.modelnamesProxyModel.index(0, 0),
                                                                     QItemSelectionModel.SelectCurrent)
            self.modelname = self.modelnamesProxyModel.index(0, 0).data()

        # keep some references
        self.batch = current_batch
        self.cellid = current_cell
        self.custom_fn = custom_fn

        # keep track of the all the models with db connections
        self.parent.db_conns.extend([self.cellsModel, self.modelnameModel])

        # setup the callbacks for viewers
        self.comboBoxBatches.currentIndexChanged.connect(self.on_batch_changed)
        self.comboBoxCells.currentIndexChanged.connect(self.on_cell_changed)
        self.listViewModelnames.selectionModel().currentChanged.connect(self.on_modelname_changed)
        # connect the filter, and the filter selector
        self.lineEditModelFilter.textChanged.connect(self.modelnamesProxyModel.setFilterFixedString)
        self.lineEditModelFilter.textChanged.connect(self.on_filter_string_changed)

        # setup the callbacks for the buttons
        self.pushButtonViewModel.clicked.connect(self.on_view_model)
        self.pushButtonViewFig.clicked.connect(self.on_view_fig)

    def update_selections(self, batch=None, cellid=None, modelname=None, custom_fn=None):
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
            modelname_index = self.modelnameModel.index(self.modelnameModel.get_index(modelname))
            proxy_index = self.modelnamesProxyModel.mapFromSource(modelname_index)
            if proxy_index.row() >= 0:
                self.modelname = modelname
                self.listViewModelnames.selectionModel().setCurrentIndex(proxy_index, QItemSelectionModel.SelectCurrent)
                self.listViewModelnames.scrollTo(proxy_index)

        if custom_fn is not None:
            self.custom_fn = custom_fn

    def load_settings(self, group_name=None):
        """Get the tabs saved selections of batch, cellid, modelname."""
        config_group = self.parent.config_group
        if group_name is not None:
            config_group = config_group + ':' + group_name

        if config_group not in self.config:
            return None, None, None, None

        batch = self.config[config_group].get(f'{self.tab_name}:batch', None)
        batch = batch if batch is None else int(batch)
        cellid = self.config[config_group].get(f'{self.tab_name}:cellid', None)
        modelname = self.config[config_group].get(f'{self.tab_name}:modelname', None)
        custom_fn = self.config[config_group].get(f'{self.tab_name}:custom_fn', None)

        return batch, cellid, modelname, custom_fn

    def get_selections(self):
        """Passes the tabs selections up to the parent for saving."""
        # sometimes during filtering, the modelname can be None
        selections = {
            f'{self.tab_name}:batch': str(self.batch),
            f'{self.tab_name}:cellid': self.cellid,
        }

        # sometimes during filtering the modelname can be None
        if self.modelname is not None:
            selections[f'{self.tab_name}:modelname'] = self.modelname

        # sometimes custom_fn can be None as well
        if self.custom_fn is not None:
            selections[f'{self.tab_name}:custom_fn'] = self.custom_fn

        return selections

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
        old_modelname = self.modelname or ''
        filter = {'batch': self.batch, 'cellid': self.cellid}
        self.modelnameModel.refresh_data(filter)

        # for qlistview, no issues with out of bounds, but we still want to disconnect and
        # reconnect so that we can ensure we emit a signal even if the index stays the same
        self.listViewModelnames.selectionModel().disconnect()

        # clear the selection to avoid some crashing from pointer/index errors
        # (I think, not having this just causes Qt to crash without raising any errors - AT)
        self.listViewModelnames.clearSelection()
        self.modelnamesProxyModel.layoutChanged.emit()
        self.modelnameModel.layoutChanged.emit()

        model_index = self.modelnamesProxyModel.index(-1, -1)
        self.listViewModelnames.selectionModel().setCurrentIndex(model_index,
                                                                 QItemSelectionModel.SelectCurrent)
        self.listViewModelnames.scrollTo(model_index)
        self.listViewModelnames.selectionModel().currentChanged.connect(self.on_modelname_changed)

        # look for old modelname in new list of modelnames
        if update_selection:
            index = self.modelnameModel.index(self.modelnameModel.get_index(old_modelname))
            proxy_index = self.modelnamesProxyModel.mapFromSource(index)
            if proxy_index.row() == -1:
                proxy_index = self.modelnamesProxyModel.index(0, 0)
            # model_index = self.modelnameModel.index(index)
            self.listViewModelnames.selectionModel().setCurrentIndex(proxy_index,
                                                                     QItemSelectionModel.SelectCurrent)
            self.listViewModelnames.scrollTo(model_index)

    def on_modelname_changed(self, current, previous):
        """Event handler for model change."""
        self.modelname = current.data()
        # self.listViewModelnames.repaint()  # TODO: figure out why not unhighlighting (BUG)
        if self.modelname is not None:
            log.info(f'Modelname changed to "{self.modelname}".')
        else:
            log.info(f'No modelname selected.')

    def on_view_model(self):
        """Event handler for view model button."""
        self.statusbar.showMessage('Loading model...', 5000)
        xfspec, ctx = xform_helper.load_model_xform(self.cellid, self.batch, self.modelname, eval_model=True)
        self.launch_model_browser(ctx, xfspec)

    def on_view_fig(self):
        """Event handler for view model button."""
        self.statusbar.showMessage('Loading figure...', 5000)

    def load_xfrom_from_folder(self, directory, eval_model=True):
        """Loads an xform/context from a directory."""
        xfspec, ctx = xforms.load_analysis(str(directory), eval_model=eval_model)
        return xfspec, ctx

    def launch_model_browser(self, ctx, xfspec):
        """Launches the model browser and keeps a reference so it's not garbage collected."""
        self.model_browser = editors.browse_xform_fit(ctx, xfspec)

    def on_filter_string_changed(self, text):
        """Reselects first after a filter change if coming from None."""
        if self.modelnamesProxyModel.rowCount() and self.modelname is None:
            proxy_index = self.modelnamesProxyModel.index(0, 0)
            self.listViewModelnames.selectionModel().setCurrentIndex(proxy_index,
                                                                     QItemSelectionModel.SelectCurrent)
            self.listViewModelnames.scrollTo(proxy_index)

    def on_action_custom_function(self):
        """Event handler for running custom function."""
        input_text, accepted = QInputDialog.getText(self, 'Custom function', 'Enter spec to custom function:',
                                                    text=self.custom_fn)
        if not accepted or not input_text:
            return

        custom_fn = lookup_fn_at(input_text)

        status_text = f'Running custom function: "{input_text}".'
        log.info(status_text)
        self.statusbar.showMessage(status_text, 2000)

        self.custom_fn = input_text
        # custom functions must implement this spec
        custom_fn(cellid=self.cellid, batch=self.batch, modelname=self.modelname)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BrowserTab()
    window .show()
    app.exec_()
