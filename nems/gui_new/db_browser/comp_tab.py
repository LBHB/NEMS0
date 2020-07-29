import logging
import sys
from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from nems.gui_new.db_browser.ui_promoted import CompModel, ListViewListModel

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
        # make some parent stuff available locally for ease
        self.statusbar = self.parent.statusbar
        self.config = self.parent.config

        # see if there's any config to load
        batch, modelname1, modelname2 = self.load_settings()

        # setup the data and models
        # self.batchModel = ListViewListModel(table_name='Results', column_name='batch')
        self.batchModel = self.parent.batchModel
        self.comboBoxBatches.setModel(self.batchModel)
        # if batched passed in, set it here
        if batch is not None:
            batch_index = self.comboBoxBatches.findText(str(batch), Qt.MatchFixedString)
            self.comboBoxBatches.setCurrentIndex(batch_index)

        current_batch = self.batchModel.index(self.comboBoxBatches.currentIndex()).data()
        batch_filter = {'batch': current_batch}
        self.modelnameModel = ListViewListModel(table_name='Results', column_name='modelname', filter=batch_filter)
        self.comboBoxModel1.setModel(self.modelnameModel)
        self.comboBoxModel2.setModel(self.modelnameModel)

        if modelname1 is not None:
            modelname1_index = self.comboBoxModel1.findText(modelname1, Qt.MatchFixedString)
            self.comboBoxModel1.setCurrentIndex(modelname1_index)
        if modelname2 is not None:
            modelname2_index = self.comboBoxModel2.findText(modelname2, Qt.MatchFixedString)
            self.comboBoxModel2.setCurrentIndex(modelname2_index)
        elif modelname1 is None:
            # set the second model to be the second value (so they're not the same)
            # could end up as same value if modelname1 is passed in as index 1, but small chance
            self.comboBoxModel2.setCurrentIndex(1)

        # keep some references
        self.batch = current_batch
        self.modelname1 = self.modelnameModel.index(self.comboBoxModel1.currentIndex()).data()
        self.modelname2 = self.modelnameModel.index(self.comboBoxModel2.currentIndex()).data()

        self.cellsModel = CompModel(
            filter1=self.modelname1,
            filter2=self.modelname2,
            table_name='Results',
            merge_on='cellid',
            comp_val='r_test',  # TODO: make this based off user input (will also need to update model.refresh_data)
            filter_on='modelname',
            filters={'batch': self.batch}
        )

        # setup the proxy filter model
        self.cellsProxyModel = QSortFilterProxyModel(self)
        self.cellsProxyModel.setSourceModel(self.cellsModel)
        self.cellsProxyModel.setFilterKeyColumn(0)
        self.cellsProxyModel.setDynamicSortFilter(True)
        self.cellsProxyModel.setFilterCaseSensitivity(Qt.CaseInsensitive)

        self.listViewCells.setModel(self.cellsProxyModel)

        self.listViewCells.setModelColumn(0)  # defaults to zero anyways, but here for clarity

        # keep track of the all the models with db connections
        self.parent.db_conns.extend([self.modelnameModel, self.cellsModel])

        # setup the plot area
        self.widgetPlot.set_labels(self.modelname1, self.modelname2, units='r_test')  # capital 'R' to see prefixes

        # setup the callbacks for the viewers
        self.comboBoxBatches.currentIndexChanged.connect(self.on_batch_changed)
        self.comboBoxModel1.currentIndexChanged.connect(self.on_modelname_changed)
        self.comboBoxModel2.currentIndexChanged.connect(self.on_modelname_changed)
        self.listViewCells.selectionModel().selectionChanged.connect(self.on_cells_changed)
        # connect the filter, and the filter selector
        self.lineEditCellFilter.textChanged.connect(self.cellsProxyModel.setFilterFixedString)
        self.lineEditCellFilter.textChanged.connect(self.on_filter_string_changed)
        # connect the buttons
        self.pushButtonSelectAll.pressed.connect(self.on_button_select_all)
        self.pushButtonPopOut.pressed.connect(self.on_button_popout)

        # select all the data
        self.listViewCells.selectAll()
        self.update_cell_count_label()

    def update_selections(self, batch=None, modelname1=None, modelname2=None):
        """Sets the inputs of batch, model1, and model2 appropriately.

        We want to avoid triggering events for all of these updates, so go through and disconnect and
        don't update the selection cascade to avoid triggering those updates until we can disable those callbacks."""
        if batch is not None:
            batch_index = self.comboBoxBatches.findText(str(batch), Qt.MatchFixedString)
            if batch_index >= 0:
                self.batch = batch
                self.comboBoxBatches.disconnect()
                self.comboBoxBatches.setCurrentIndex(batch_index)
                self.on_batch_changed(batch_index, update_selection=True if modelname1 is None else False)
                self.comboBoxBatches.currentIndexChanged.connect(self.on_batch_changed)

        if modelname1 is not None:
            modelname1_index = self.comboBoxModel1.findText(modelname1, Qt.MatchFixedString)
            if modelname1_index >= 0:
                self.modelname1 = modelname1
                self.comboBoxModel1.disconnect()
                self.comboBoxModel1.setCurrentIndex(modelname1_index)
                if modelname2 is None:
                    self.on_modelname_changed(None, update_selection=True)
                self.comboBoxModel1.currentIndexChanged.connect(self.on_modelname_changed)

        if modelname2 is not None:
            modelname2_index = self.comboBoxModel2.findText(modelname2, Qt.MatchFixedString)
            if modelname2_index >= 0:
                self.modelname2 = modelname2
                self.comboBoxModel2.disconnect()
                self.comboBoxModel2.setCurrentIndex(modelname2_index)
                self.on_modelname_changed(None, update_selection=True)
                self.comboBoxModel2.currentIndexChanged.connect(self.on_modelname_changed)

    def load_settings(self, group_name=None):
        """Get the tabs saved selections of batch, model1, and model2."""
        config_group = self.parent.config_group
        if group_name is not None:
            config_group = config_group + ':' + group_name

        if config_group not in self.config:
            return None, None, None

        batch = self.config[config_group].get(f'{self.tab_name}:batch', None)
        batch = batch if batch is None else int(batch)
        modelname1 = self.config[config_group].get(f'{self.tab_name}:modelname1', None)
        modelname2 = self.config[config_group].get(f'{self.tab_name}:modelname2', None)

        return batch, modelname1, modelname2

    def get_selections(self):
        """Passes the tabs selections up to the parent for saving."""
        selections = {
            f'{self.tab_name}:batch': str(self.batch),
            f'{self.tab_name}:modelname1': self.modelname1,
            f'{self.tab_name}:modelname2': self.modelname2,
        }

        return selections

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
        # two events (and reconnect the second so that both get updated)
        self.comboBoxModel2.currentIndexChanged.connect(self.on_modelname_changed)

        # self.cellsModel.layoutChanged.emit()
        self.cellsProxyModel.layoutChanged.emit()

        # look for old cell in new list of cells
        if update_selection:
            index1 = self.comboBoxModel1.findText(old_modelname1, Qt.MatchFixedString)
            index2 = self.comboBoxModel2.findText(old_modelname2, Qt.MatchFixedString)
            if index1 == -1:
                index1 = 0
            if index2 == -1:
                index2 = 1
        else:
            index1 = 0
            index2 = 1

        self.comboBoxModel1.setCurrentIndex(index1)
        self.comboBoxModel2.setCurrentIndex(index2)

        # do a manual redraw since slow to update sometimes, not sure why
        self.comboBoxModel1.update()
        self.comboBoxModel2.update()
        self.comboBoxModel1.currentIndexChanged.connect(self.on_modelname_changed)

    def on_modelname_changed(self, index, update_selection=True):
        """Event handler for model change."""
        self.modelname1 = self.modelnameModel.index(self.comboBoxModel1.currentIndex()).data()
        self.modelname2 = self.modelnameModel.index(self.comboBoxModel2.currentIndex()).data()
        log.info('Modelnames changed, loading cellids.')

        # here we don't keep track of the old cellids because since it's extended selection, it would be expensive
        # to try to search for all the matching cellids
        filter = {'batch': self.batch}
        self.cellsModel.refresh_data(
            filter1=self.modelname1,
            filter2=self.modelname2,
            filters=filter)

        # clear the selection to avoid some crashing from pointer/index errors
        # (I think, not having this just causes Qt to crash without raising any errors - AT)
        self.listViewCells.clearSelection()

        self.cellsProxyModel.layoutChanged.emit()
        self.cellsModel.layoutChanged.emit()

        self.listViewCells.selectAll()
        self.widgetPlot.set_labels(self.modelname1, self.modelname2)
        self.on_cells_changed(None, None)
        self.update_cell_count_label()

    def on_cells_changed(self, selected, deselected):
        """Event handler for model change."""
        self.cellids = [self.cellsProxyModel.index(i.row(), 0).data() for i in self.listViewCells.selectedIndexes()]

        rows = [i.row() for i in self.listViewCells.selectedIndexes()]
        if self.cellsModel.np_points is None:
            self.widgetPlot.clear_data()
        else:
            self.widgetPlot.update_data(y=self.cellsModel.np_points[0][rows], x=self.cellsModel.np_points[1][rows],
                                        labels=self.cellsModel.np_labels[rows])

    def update_cell_count_label(self):
        self.labelCellCount.setText(f'Cell IDs (n={self.cellsProxyModel.rowCount()}):')

    def on_filter_string_changed(self, text):
        """Reselects all after a filter change."""
        self.listViewCells.selectAll()
        self.update_cell_count_label()

    def on_button_select_all(self):
        """Event handler for hitting 'Select All'."""
        self.listViewCells.selectAll()

    def on_button_popout(self):
        """Event handler for hitting 'Select All'."""
        # make a dialog and embed the new plot widget
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose)  # this ensure that the popup will close when the main app exits
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        dialog.setLayout(layout)
        # dialog.setGeometry(100, 100, 520, 500)

        plot_widget = self.widgetPlot.get_copy()
        dialog.setWindowTitle(f'"{plot_widget.left_label}" vs "{plot_widget.bottom_label}"')
        layout.addWidget(plot_widget)
        dialog.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CompTab()
    window .show()
    app.exec_()
