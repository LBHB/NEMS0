import sys
import pandas as pd
from pathlib import Path
from functools import partial

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5 import uic

from nems import db as nd
from nems.utils import simple_search, load_settings, save_settings
import nems.xform_helper as xhelp

qt_creator_file = Path(r'ui') / 'tab_analysis.ui'
Ui_Widget, QtBaseClass = uic.loadUiType(qt_creator_file)

Qt = QtCore.Qt
import db_test


class pandasModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        QtCore.QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


class AnalysisTab(QtBaseClass, Ui_Widget):
    def __init__(self, parent=None):
        super(AnalysisTab, self).__init__(parent)
        self.setupUi(self)

        self.tab_name = 'analysis_tab'

    def init_models(self):
        self.statusbar = self.parent.statusbar
        self.config = self.parent.config

        d = load_settings('analysis')
        self.current_analysis = d.get('analysis', '')
        self.lastbatch = ''
        self.last_loaded = [None, None, None, None]
        self.ex = None

        self.all_models = []
        self.all_cellids = []
        self.batch_data = []
        self.analysis_data = []

        self.refresh_batches()

        self.pushUpdate.clicked.connect(self.refresh_lists)
        #self.listCellid.selectionChanged.connect(self.update_model_info)
        #self.listModelname.selectionChanged.connect(self.update_model_info)

        self.comboBatch.currentIndexChanged.connect(self.reload_models)
        self.comboAnalysis.currentIndexChanged.connect(self.analysis_update)
        self.pushScatter.clicked.connect(partial(self.analysis, 'scatter'))
        self.pushBar.clicked.connect(partial(self.analysis, 'bar'))
        self.pushPareto.clicked.connect(partial(self.analysis, 'pareto'))
        self.pushView.clicked.connect(partial(self.analysis, 'view'))



    def refresh_batches(self):

        sql = "SELECT * FROM Analysis order by id"
        self.analysis_data = nd.pd_query(sql)
        model = QtGui.QStandardItemModel()
        for i in self.analysis_data['name'].to_list():
            item = QtGui.QStandardItem(i)
            model.appendRow(item)
        self.comboAnalysis.setModel(model)
        index = self.comboAnalysis.findText(self.current_analysis, QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.comboAnalysis.setCurrentIndex(index)

        sql = "SELECT DISTINCT batch FROM Batches order by batch"
        self.batch_data = nd.pd_query(sql)
        model = QtGui.QStandardItemModel()
        for i in self.batch_data['batch'].to_list():
            item = QtGui.QStandardItem(i)
            model.appendRow(item)
        self.comboBatch.setModel(model)

        self.analysis_update()

    def reload_models(self):
        t = self.comboBatch.currentText()
        sql = f"SELECT DISTINCT modelname FROM Results WHERE batch={t}"
        data = nd.pd_query(sql)
        self.all_models = data['modelname'].to_list()
        sql = f"SELECT DISTINCT cellid FROM Batches WHERE batch={t}"
        data = nd.pd_query(sql)
        self.all_cellids = data['cellid'].to_list()
        self.lastbatch = t

        self.labelBatchName.setText(str(t))

    def analysis_update(self):
        self.current_analysis = self.comboAnalysis.currentText()
        analysis_info = self.analysis_data[
            self.analysis_data.name == self.current_analysis].reset_index()

        if analysis_info['model_search'][0] is not None:
            self.lineModelname.setText(analysis_info['model_search'][0])
        if analysis_info['cell_search'][0] is not None:
            self.lineCellid.setText(analysis_info['cell_search'][0])

        current_batch = analysis_info['batch'].values[0].split(":")[0]
        if current_batch != self.comboBatch.currentText():
            index = self.comboBatch.findText(current_batch, QtCore.Qt.MatchFixedString)
            if index >= 0:
                self.comboBatch.setCurrentIndex(index)
                self.refresh_lists()
        save_settings('analysis', {'analysis': self.current_analysis})

    def refresh_lists(self):

        t = self.comboBatch.currentText()
        if t != self.lastbatch:
            self.reload_models()

        cell_search = self.lineCellid.text()
        refined_list = simple_search(cell_search, self.all_cellids)

        model = QtGui.QStandardItemModel()
        for i in refined_list:
            item = QtGui.QStandardItem(i)
            model.appendRow(item)
        self.listCellid.setModel(model)
        self.listCellid.selectionModel().selectionChanged.connect(self.update_model_info)

        model_search = self.lineModelname.text()
        refined_model_list = simple_search(model_search, self.all_models)

        model = QtGui.QStandardItemModel()
        for i in refined_model_list:
            item = QtGui.QStandardItem(i)
            model.appendRow(item)
        #d = pandasModel(data)
        #self.tableModelname.setModel(d)

        self.listModelname.setModel(model)
        self.listModelname.selectionModel().selectionChanged.connect(self.update_model_info)

        # save cell and model search strings
        t = self.comboAnalysis.currentText()
        analysis_info = self.analysis_data[self.analysis_data.name == t].reset_index()
        sql = f"UPDATE Analysis set cell_search='{cell_search}',"+\
            f" model_search='{model_search}'"+\
            f" WHERE id={analysis_info['id'][0]}"
        nd.sql_command(sql)

    def get_selected(self):
        batch = self.comboBatch.currentText()

        _idxs = self.listCellid.selectedIndexes()
        if len(_idxs) == 0:
            # if none selected assume all
            self.listCellid.selectAll()
            _idxs = self.listCellid.selectedIndexes()
        selectedCellid = [self.listCellid.model().item(i.row()).text()
                          for i in _idxs]

        _idxs = self.listModelname.selectedIndexes()
        if len(_idxs) == 0:
            # if none selected assume all
            self.listModelname.selectAll()
            _idxs = self.listModelname.selectedIndexes()
        selectedModelname = [self.listModelname.model().item(i.row()).text()
                         for i in _idxs]

        return batch, selectedCellid, selectedModelname


    def get_current_selection(self):

        batch, selectedCellid, selectedModelname = self.get_selected()
        print('cellid(s)', selectedCellid)
        print('modelnames(s)', selectedModelname)
        try:
            cellid = selectedCellid[0]
            modelname = selectedModelname[0]
        except AttributeError:
            print("You must select a cellid and modelname first")
            return None, None

        print("Loading {},{},{}".format(batch,cellid,modelname))

        if (self.last_loaded[0]==cellid and self.last_loaded[1]==modelname and
            self.last_loaded[2] == batch):
            xf = self.last_loaded[3]
            ctx = self.last_loaded[4]
        else:
            xf, ctx = xhelp.load_model_xform(cellid, batch, modelname, eval_model=True)
            self.last_loaded = [cellid, modelname, batch, xf, ctx]

        return xf, ctx

    def update_model_info(self):
        batch, selectedCellid, selectedModelname = self.get_selected()
        self.labelModelInfo.setText(selectedCellid[0] + ": " + selectedModelname[0])


    def run_custom(self):

        analysis_name = self.lineAnalysis.text()

        self.analysis(analysis_name)


    def analysis(self, analysis_name):
        batch, selectedCellid, selectedModelname = self.get_selected()

        if analysis_name == 'pareto':
            from nems_lbhb.plots import model_comp_pareto
            model_comp_pareto(selectedModelname, batch=int(batch),
                              goodcells=selectedCellid)
        elif analysis_name == 'view':
            import nems.gui.editors as editor
            xf, ctx = self.get_current_selection()
            self.ex = editor.EditorWidget(modelspec=ctx['modelspec'], rec=ctx['val'], xfspec=xf,
                                          ctx=ctx, parent=self)
            self.ex.show()
        else:
            print('Unknown analysis_name')


def main():
    app = QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()