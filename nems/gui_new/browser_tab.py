from pathlib import Path
import logging
import sys
from configparser import ConfigParser, DuplicateSectionError

from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from nems import db, xforms, get_setting
from nems.gui import editors

from nems.gui_new.list_test import ListViewModel

log = logging.getLogger(__name__)

# read and parse the UI file to generate the GUI
qt_creator_file = Path(r'ui\tab_browser.ui')
Ui_Widget, QtBaseClass = uic.loadUiType(qt_creator_file)


class ListViewModel(QAbstractListModel):
    def __init__(self,
                 table_name: str,
                 column_name: str,
                 filter=None,
                 *args,
                 **kwargs
                 ):
        super(ListViewModel, self).__init__(*args, **kwargs)

        self.table_name = table_name
        self.column_name = column_name

        self.data = None

        self.init_db()
        self.refresh_data(filters=filter)

    def data(self, index, role):
        if role == Qt.DisplayRole:
            text = self.data[index.row()][0]
            return text

    def rowCount(self, index):
        return len(self.data)

    def init_db(self):
        self.engine = db.Engine()
        table = db.Tables()[self.table_name]
        self.column = getattr(table, self.column_name)
        self.session = db.sessionmaker(bind=self.engine)()

    def close_db(self):
        self.session.close()
        self.engine.dispose()

    def refresh_data(self, filters: dict = None):
        """Reads the data from the database, optionally filtering.

        :param filters: Dict of column: filter.
        :return:
        """
        query = (self.session
                 .query(self.column)
                 .distinct()
                 )

        if filters is not None:
            query = query.filter_by(**filters)

        query = query.order_by(self.column)
        data = query.all()

        self.data = data

    def get_index(self, value) -> int:
        """Searches for a value in the data, returns the index"""
        flat = [i[0] for i in self.data]
        try:
            return flat.index(value)
        except ValueError:
            return -1


class BrowserTab(QtBaseClass, Ui_Widget):

    def __init__(self, parent=None):
        super(BrowserTab, self).__init__(parent)
        self.setupUi(self)

        # load settings
        batch, cellid, modelname = self.load_settings()

        # setup the data and models
        self.batchModel = ListViewModel(table_name='Results', column_name='batch')
        self.comboBoxBatches.setModel(self.batchModel)

        current_batch = self.batchModel.data[self.comboBoxBatches.currentIndex()][0]
        cell_filter = {'batch': current_batch}
        self.cellsModel = ListViewModel(table_name='Results', column_name='cellid', filter=cell_filter)
        self.comboBoxCells.setModel(self.cellsModel)

        current_cell = self.cellsModel.data[self.comboBoxCells.currentIndex()][0]
        model_filter = {'batch': current_batch, 'cellid': current_cell}
        self.modelnameModel = ListViewModel(table_name='Results', column_name='modelname', filter=model_filter)
        self.listViewModelnames.setModel(self.modelnameModel)
        # select first entry
        self.listViewModelnames.selectionModel().setCurrentIndex(self.modelnameModel.index(0),
                                                                 QItemSelectionModel.SelectCurrent)

        # keep some references
        self.batch = current_batch
        self.cellid = current_cell
        self.modelname = self.modelnameModel.index(0).data()

        # keep track of the all the models with db connections
        self.conns = [self.batchModel, self.cellsModel, self.modelnameModel]

        # setup the callbacks for viewers
        self.comboBoxBatches.currentIndexChanged.connect(self.on_batch_changed)
        self.comboBoxCells.currentIndexChanged.connect(self.on_cell_changed)
        self.listViewModelnames.selectionModel().currentChanged.connect(self.on_modelname_changed)

        # setup the callbacks for the buttons
        self.pushButtonViewModel.clicked.connect(self.on_view_model)

        # update inputs
        self.update_selections(*self.load_settings())

    def closeEvent(self, event):
        """Catch close event in order to close db connections and save selections."""
        for model in self.conns:
            model.close_db()

        self.save_settings()
        event.accept()

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
                self.listViewModelnames.selectionModel().setCurrentIndex(self.modelnameModel.index(modelname_index),
                                                                         QItemSelectionModel.SelectCurrent)

    def get_setting_sections(self):
        """Gets a list of the sections in the settings."""
        config_file = Path(get_setting('SAVED_SETTINGS_PATH')) / 'gui.ini'
        config = ConfigParser()
        config.read(config_file)

        return [section.lstrip('db_browser:') for section in config.sections() if section.startswith('db_browser:')]

    def load_settings(self, group_name=None):
        """Loads saved selections of batch, cellid, modelname."""
        config_file = Path(get_setting('SAVED_SETTINGS_PATH')) / 'gui.ini'
        config = ConfigParser()
        config.read(config_file)

        config_group = 'db_browser'
        if group_name is not None:
            config_group = config_group + ':' + group_name

        if config_group not in config:
            return None, None, None

        batch = int(config[config_group]['batch'])
        cellid = config[config_group]['cellid']
        modelname = config[config_group]['modelname']

        return batch, cellid, modelname

    def save_settings(self, group_name=None):
        """Saves the current selections."""
        config_file = Path(get_setting('SAVED_SETTINGS_PATH')) / 'gui.ini'
        config = ConfigParser()
        config.read(config_file)

        config_group = 'db_browser'
        if group_name is not None:
            config_group = config_group + ':' + group_name

        # add section if not present
        try:
            config.add_section(config_group)
        except DuplicateSectionError:
            pass

        config.set(config_group, 'batch', str(self.batch))
        config.set(config_group, 'cellid', self.cellid)
        config.set(config_group, 'modelname', self.modelname)

        with open(config_file, 'w') as cf:
            config.write(cf)

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

    def on_action_save_selections(self, *args, text=''):
        """Event handler for saving selections."""
        input_text, accepted = QInputDialog.getText(self, 'Selection name', 'Enter selection name:', text=text)
        if not accepted or not input_text:
            return

        existing = self.get_setting_sections()
        if input_text in existing:
            ret = QMessageBox.warning(self, 'Save', 'Name exists, overwrite?',
                                      QMessageBox.Save | QMessageBox.Cancel)
            if ret == QMessageBox.Cancel:
                self.on_action_save_selections(text=input_text)
        else:
            self.save_settings(group_name=input_text)

        status_text = f'Saved selections as: "{input_text}".'
        log.info(status_text)
        self.statusbar.showMessage(status_text, 1000)

    def on_action_load_selections(self):
        """Event handler for loading selections."""
        existing = self.get_setting_sections()
        group_name, accepted = QInputDialog.getItem(self, 'Selection name', 'Select settings to load:', existing,
                                                    editable=False)
        if not accepted:
            return

        self.update_selections(*self.load_settings(group_name=group_name))

        status_text = f'Loaded saved selections: "{group_name}".'
        log.info(status_text)
        self.statusbar.showMessage(status_text, 1000)

    def on_batch_changed(self, index, update_selection=True):
        """Event handler for batch selection.

        Starts cascade of updating the proper UI elements. Looks to see if current cell exists
        in new batch and if so sets it, otherwise resets index to 0.
        """
        self.batch = self.batchModel.data[index][0]
        log.info(f'Batch changed to "{self.batch}", loading cellids.')
        filter = {'batch': self.batch}

        old_cellid = self.comboBoxCells.currentText()
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

    def on_cell_changed(self, index, update_selection=True):
        """Event handler for cellid change."""
        self.cellid = self.cellsModel.data[index][0]
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
