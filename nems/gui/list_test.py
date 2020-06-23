import sys
from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from nems import db

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

# read and parse the UI file to generate the GUI
qt_creator_file = Path(r'ui/test_listview.ui')
Ui_MainWindow, QtBaseClass = uic.loadUiType(qt_creator_file)


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


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

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

        # keep track of the all the models with db connections
        self.conns = [self.batchModel, self.cellsModel, self.modelnameModel]

        # setup the callbacks
        self.comboBoxBatches.currentIndexChanged.connect(self.batch_changed)
        self.comboBoxCells.currentIndexChanged.connect(self.cell_changed)
        self.listViewModelnames.selectionModel().currentChanged.connect(self.modelname_changed)

    def closeEvent(self, event):
        for model in self.conns:
            model.close_db()

        event.accept()

    def batch_changed(self, index):
        """Event handler for batch selection.

        Starts cascade of updating the proper UI elements. Looks to see if current cell exists
        in new batch and if so sets it, otherwise resets index to 0.
        """
        new_batch = self.batchModel.data[index][0]
        filter = {'batch': new_batch}

        old_cellid = self.comboBoxCells.currentText()
        self.cellsModel.refresh_data(filter)

        # need to ensure the current index is in bounds before updating the combobox
        # but temporarily don't emit a signal since we'll send it below again
        self.comboBoxCells.disconnect()
        self.comboBoxCells.setCurrentIndex(-1)
        self.comboBoxCells.currentIndexChanged.connect(self.cell_changed)

        self.cellsModel.layoutChanged.emit()

        # look for old cell in new list of cells
        index = self.comboBoxCells.findText(old_cellid, Qt.MatchFixedString)
        if index == -1:
            index = 0
        self.comboBoxCells.setCurrentIndex(index)

    def cell_changed(self, index):
        """Event handler for cellid change."""
        new_cellid = self.cellsModel.data[index][0]
        batch = self.batchModel.data[self.comboBoxBatches.currentIndex()][0]

        # qcombobox and qlistview work a bit different in terms of accessing data
        old_modelname = self.listViewModelnames.selectedIndexes()[0].data()
        filter = {'batch': batch, 'cellid': new_cellid}
        self.modelnameModel.refresh_data(filter)

        self.modelnameModel.layoutChanged.emit()

        # look for old modelname in new list of modelnames
        index = self.modelnameModel.get_index(old_modelname)
        if index == -1:
            index = 0
        self.listViewModelnames.selectionModel().setCurrentIndex(self.modelnameModel.index(index),
                                                                 QItemSelectionModel.SelectCurrent)

        print('changed: ', new_cellid)

    def modelname_changed(self, current, previous):
        """Event handler for model change."""
        print(current.data())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window .show()
    app.exec_()
