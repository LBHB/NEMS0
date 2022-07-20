import sys
import numpy as np
import pandas as pd
import pandas.io.sql as psql
from pathlib import Path
from functools import partial
import itertools
import seaborn as sns
import json

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5 import uic

from nems0 import db as nd
from nems.utils import lookup_fn_at, simple_search #, load_settings, save_settings
import nems.xform_helper as xhelp
from nems.gui_new.db_browser import model_browser

from nems.db import Session, Tables, get_batch_cells
import matplotlib.pyplot as plt

qt_creator_file = Path(r'ui') / 'tab_analysis.ui'
Ui_Widget, QtBaseClass = uic.loadUiType(qt_creator_file)

Qt = QtCore.Qt



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

        self.current_analysis = ''
        self.lastbatch = ''
        self.last_loaded = [None, None, None, None]
        self.ex = None
        self.mb = None

        self.all_models = []
        self.all_cellids = []
        self.batch_data = []
        self.analysis_data = []

        self.load_local_settings()
        self.refresh_batches()

        self.pushUpdate.clicked.connect(self.refresh_lists)
        #self.listCellid.selectionChanged.connect(self.update_model_info)
        #self.listModelname.selectionChanged.connect(self.update_model_info)

        self.comboBatch.currentIndexChanged.connect(self.reload_models)
        self.comboAnalysis.currentIndexChanged.connect(self.analysis_update)
        self.pushScatter.clicked.connect(self.scatter)
        self.pushLineComp.clicked.connect(self.line_comp)
        self.pushBar.clicked.connect(self.bar)
        self.pushPareto.clicked.connect(partial(self.analysis, 'pareto'))
        self.pushView.clicked.connect(self.view)
        self.pushViewNew.clicked.connect(partial(self.view, new=True))
        self.pushCustomSingle.clicked.connect(self.run_custom_single),
        self.pushCustomGroup.clicked.connect(self.run_custom_group)

    def load_local_settings(self, group_name=None):
        """Get the tabs saved selections of batch, cellid, modelname."""
        config_group = self.parent.config_group
        if group_name is not None:
            config_group = config_group + ':' + group_name

        if config_group not in self.config:
            return
        self.current_analysis = self.config[config_group].get(f'{self.tab_name}:analysis', None)
        self.lineCustomSingle.setText(self.config[config_group].get(f'{self.tab_name}:custom_single', ""))
        self.lineCustomGroup.setText(self.config[config_group].get(f'{self.tab_name}:custom_group', ""))

    def get_selections(self):
        """Passes the tabs selections up to the parent for saving."""
        # sometimes during filtering, the modelname can be None
        selections = {
            f'{self.tab_name}:analysis': self.current_analysis,
            f'{self.tab_name}:custom_single': self.lineCustomSingle.text() ,
            f'{self.tab_name}:custom_group': self.lineCustomGroup.text() 
        }

        return selections

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
        from nems_web.utilities.ModelFinder import ModelFinder

        t = self.comboBatch.currentText()
        session = Session()
        Analysis = Tables()['Analysis']
 
        modeltree = (
                session.query(Analysis.modeltree)
                .filter(Analysis.name == self.current_analysis)
                .first()
                )
        modelextras = (
                session.query(Analysis.model_extras)
                .filter(Analysis.name == self.current_analysis)
                .first()
                )
        # Pass modeltree string from Analysis to a ModelFinder constructor,
        # which will use a series of internal methods to convert the tree string
        # to a list of model names.
        # Then add any additional models specified in extraModels, and add
        # model_lists from extraAnalyses.
        if modeltree and modeltree[0]:
            #model_list = _get_models(modeltree[0])
            load, mod, fit = json.loads(modeltree[0])
            loader = ModelFinder(load).modellist
            model = ModelFinder(mod).modellist
            fitter = ModelFinder(fit).modellist
            combined = itertools.product(loader, model, fitter)
            model_list = ['_'.join(m) for m in combined]
            extraModels = [s.strip("\"\n").replace("\\n","") for s in modelextras[0].split(',')]
            model_list.extend(extraModels)
        else:
            model_list=[]
        self.all_models = model_list

        #sql = f"SELECT DISTINCT modelname FROM Results WHERE batch={t}"
        #data = nd.pd_query(sql)
        #self.all_models = data['modelname'].to_list()
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
        #self.save_local_settings()

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

    def run_custom_single(self):

        analysis_name = self.lineCustomSingle.text()
        print('Custom analysis: ',analysis_name)
        batch, selectedCellid, selectedModelname = self.get_selected()
        try:
            f = lookup_fn_at(analysis_name)
            xf, ctx = self.get_current_selection()
            f(**ctx)
            plt.show()
        except:
            print('Unknown/incompatible analysis_name')
        #self.save_local_settings()

    def run_custom_group(self):

        analysis_name = self.lineCustomGroup.text()
        print('Custom group analysis: ',analysis_name)
        batch, selectedCellid, selectedModelname = self.get_selected()
        try:
            f = lookup_fn_at(analysis_name)
            f(selectedModelname, batch=int(batch), goodcells=selectedCellid)
        except:
            print('Unknown/incompatible analysis_name')
        #self.save_local_settings()

    def get_sum_data(self, measure):

        import nems_db.plot_helpers as dbp
        batch, selectedCellid, selectedModelname = self.get_selected()

        plot = dbp.plot_filtered_batch(batch, selectedModelname, measure, 'Scatter', only_fair=True, include_outliers=False, display=False)
        data = plot.data

        cellids = data.index.levels[1].tolist()
        cidx = [c for c in selectedCellid if c in cellids]
        if len(cidx)==0:
            cidx = cellids
        data = data.loc[pd.IndexSlice[:, cidx], :]
        return data

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
            print('Unknown/incompatible analysis_name')

    def view(self, new=False):
        batch, selectedCellid, selectedModelname = self.get_selected()
        xf, ctx = self.get_current_selection()

        if new:
            self.mb = model_browser.MainWindow(ctx=ctx, xfspec=xf)
        else:
            import nems.gui.editors as editor
            self.ex = editor.EditorWidget(modelspec=ctx['modelspec'], rec=ctx['val'], xfspec=xf,
                                          ctx=ctx, parent=self)
            self.ex.show()

    def scatter(self):

        measure=['r_test']
        data = self.get_sum_data(measure[0])
        modelnames = data.index.levels[0].tolist()


        for pair in list(itertools.combinations(modelnames,2)):

            modelX = pair[0]
            modelY = pair[1]

            dataX = data.loc[modelX]
            dataY = data.loc[modelY]

            cells = []
            cellsX = list(set(dataX.index.values.tolist()))
            cellsY = list(set(dataY.index.values.tolist()))
            cells = cellsX
            x_mean = np.mean(dataX[measure[0]])
            x_median = np.median(dataX[measure[0]])
            y_mean = np.mean(dataY[measure[0]])
            y_median = np.median(dataY[measure[0]])

            x_label = (
                    "{0}\nmean: {1:5.4f}, median: {2:5.4f}"
                    .format(modelX, x_mean, x_median)
                    )
            y_label = (
                    "{0}\nmean: {1:5.4f}, median: {2:5.4f}"
                    .format(modelY, y_mean, y_median)
                    )

            data = {
                    'x_values':dataX[measure[0]],
                    'y_values':dataY[measure[0]],
                    'cellid':cells,
                    }
            f,ax=plt.subplots(1,1)
            ax.plot([0,1],[0,1],'k--',linewidth=0.5)
            ax.scatter(data['x_values'], data['y_values'], s=4, color='k')
            ax.set_aspect('equal','box')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            plt.show()


    def line_comp(self):
        measure=['r_test']
        data = self.get_sum_data(measure[0])
        modelnames = data.index.levels[0].tolist()

        f, ax = plt.subplots(1, 1)
        for m in modelnames:

            dataX = data.loc[m]

            cells = list(set(dataX.index.values.tolist()))
            x_mean = np.mean(dataX[measure[0]])
            x_median = np.median(dataX[measure[0]])

            x_label = (
                "{0}\nmean: {1:5.4f}, median: {2:5.4f}".format(m, x_mean, x_median)
            )

            ax.plot(dataX[measure[0]], label=x_label)
        plt.xticks(rotation=90)
        ax.set_xlabel('unit')
        ax.set_ylabel(measure[0])
        ax.legend(frameon=False)
        plt.show()

    def bar(self):
        measure=['r_test']
        data = self.get_sum_data(measure[0])
        data.loc[:,measure[0]] = data[measure[0]].astype(float)
        data = data.groupby('modelname').mean()


        #data = data.reset_index()
        modelnames = [m.replace("-", "\n") for m in data.index]

        f, ax = plt.subplots(1, 1)
        ax.bar(modelnames, data[measure[0]], color='red')
        #ax.set_xticklabels(modelnames)

        for i, lbl in enumerate(data.index):
            plt.text(i, data.loc[lbl, measure[0]]+0.01, f"{data.loc[lbl, measure[0]]:.3f}", ha='center', va='bottom')

        ax.set_xlabel('Model')
        ax.set_ylabel(measure[0])
        plt.show()


def main():
    app = QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()
