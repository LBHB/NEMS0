#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:05:34 2018

@author: svd
"""

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import os.path
import importlib

import PyQt5.QtCore as qc
import PyQt5.QtGui as qg
import PyQt5.QtWidgets as qw

import nems.db as nd
import nems.plots.api as nplt
import nems.xform_helper as xhelp
import nems.epoch as ep
from nems.utils import find_common
import pandas as pd
import scipy.ndimage.filters as sf
import nems_lbhb.plots as lplt
import nems.gui.recording_browser as browser
import nems.gui.editors as editor
import nems.gui.model_comparison as comparison
from configparser import ConfigParser
import nems

configfile = os.path.join(nems.get_setting('SAVED_SETTINGS_PATH') + '/gui.ini')

def load_settings(m):

    config = ConfigParser()
    try:
        config.read(configfile)
        m.batchLE.setText(config.get('db_browser_last', 'batch'))
        m.cellLE.setText(config.get('db_browser_last', 'cells'))
        m.modelLE.setText(config.get('db_browser_last', 'models'))

        geostr = config.get('db_browser_last', 'geometry')
        g=[int(x) for x in geostr.split(",")]
        m.setGeometry(*g)

    except:
        with open(configfile, 'w') as f:
            config.write(f)
        save_settings(m)

def save_settings(m):

    config = ConfigParser()
    config.read(configfile)

    batch = m.batchLE.text()
    cellmask = m.cellLE.text()
    modelmask = m.modelLE.text()
    rect = m.frameGeometry().getRect()
    geostr = ",".join([str(x) for x in rect])
    #print(geostr)
    try:
        # Create non-existent section
        config.add_section('db_browser_last')
    except:
        pass

    config.set('db_browser_last', 'batch', batch)
    config.set('db_browser_last', 'cells', cellmask)
    config.set('db_browser_last', 'models', modelmask)
    config.set('db_browser_last', 'geometry', geostr)

    with open(configfile, 'w') as f:
        config.write(f)


class PandasModel(qc.QAbstractTableModel):
    """
    Thanks to this link!
    https://github.com/eyllanesc/stackoverflow/tree/master/questions/44603119
    """
    def __init__(self, df = pd.DataFrame(), parent=None):
        qc.QAbstractTableModel.__init__(self, parent=parent)
        self._df = df

    def headerData(self, section, orientation, role=qc.Qt.DisplayRole):
        if role != qc.Qt.DisplayRole:
            return qc.QVariant()

        if orientation == qc.Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return qc.QVariant()
        elif orientation == qc.Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return qc.QVariant()

    def data(self, index, role=qc.Qt.DisplayRole):
        if role != qc.Qt.DisplayRole:
            return qc.QVariant()

        if not index.isValid():
            return qc.QVariant()

        return qc.QVariant(str(self._df.ix[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=qc.QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=qc.QModelIndex()):
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending= order == qc.Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()


class model_browser(qw.QWidget):
    """
    For a given batch, list all cellids and modelnames matching in
    NarfResults. Clicking view will call view_model_recording for the
    currently selected model.

    """
    def __init__(self, batch=289,
                 cell_search_string="",
                 model_search_string="ozgf.fs100.ch18",
                 parent=None):
        qw.QWidget.__init__(self, parent=None)

        # Keep reference to opened recordings so they don't
        # get garbage-collected
        self._cached_windows = []

        # parameters for caching last model loaded
        self.last_loaded=['x','x',0]
        self.recname='val'

        # selection parameters
        self.batch = batch

        # main layout
        vLayout2 = qw.QVBoxLayout(self)
        hLayout1 = qw.QHBoxLayout()

        self.cells = qw.QListWidget(self)
        self.cells.setMaximumSize(qc.QSize(130, 1000))
        self.cells.setSelectionMode(qw.QAbstractItemView.ExtendedSelection)
        self.models = qw.QListWidget(self)
        self.models.setSelectionMode(qw.QAbstractItemView.ExtendedSelection)
        self.data_model = PandasModel(parent=self)
        """
        self.data_table = qw.QTableView(self)
        #self.data_table.setModel(self.data_model)
        """
        hLayout1.addWidget(self.cells)
        hLayout1.addWidget(self.models)
        #hLayout1.addWidget(self.data_table)

        hLayout2 = qw.QHBoxLayout()

        formLayout = qw.QFormLayout(self)
        batchlabel = qw.QLabel(self)
        batchlabel.setText('Batch:')
        batchlabel.setMaximumWidth(200)
        self.batchLE = qw.QLineEdit(self)
        self.batchLE.setText(str(batch))
        self.batchLE.returnPressed.connect(self.update_widgets)
        self.batchLE.setMaximumWidth(200)
        formLayout.addRow(batchlabel, self.batchLE)

        celllabel = qw.QLabel(self)
        celllabel.setText('CellID:')
        celllabel.setMaximumWidth(200)
        self.cellLE = qw.QLineEdit(self)
        self.cellLE.setText(cell_search_string)
        self.cellLE.returnPressed.connect(self.update_widgets)
        self.cellLE.setMaximumWidth(200)
        formLayout.addRow(celllabel,self.cellLE)

        modellabel = qw.QLabel(self)
        modellabel.setMaximumWidth(200)
        modellabel.setText('Modelname:')
        self.modelLE = qw.QLineEdit(self)
        self.modelLE.setText(model_search_string)
        self.modelLE.returnPressed.connect(self.update_widgets)
        # self.modelLE.setMaximumWidth(500)
        formLayout.addRow(modellabel,self.modelLE)

        vLayout = qw.QVBoxLayout()
        self.updateBtn = qw.QPushButton("Re-import", self)
        self.updateBtn.clicked.connect(self.reimport_libs)
        vLayout.addWidget(self.updateBtn)
        self.updateBtn.setMaximumWidth(150)

        self.sumBtn = qw.QPushButton("Pred sum", self)
        self.sumBtn.clicked.connect(self.run_fun)
        vLayout.addWidget(self.sumBtn)
        self.sumBtn.setMaximumWidth(150)

        self.viewBtn = qw.QPushButton("View recording", self)
        self.viewBtn.clicked.connect(self.view_recording)
        vLayout.addWidget(self.viewBtn)
        self.viewBtn.setMaximumWidth(150)

        self.modelBtn = qw.QPushButton("View model", self)
        self.modelBtn.clicked.connect(self.view_model)
        vLayout.addWidget(self.modelBtn)
        self.modelBtn.setMaximumWidth(150)

        self.compareBtn = qw.QPushButton("Compare models", self)
        self.compareBtn.clicked.connect(self.compare_models)
        vLayout.addWidget(self.compareBtn)
        self.compareBtn.setMaximumWidth(150)

        hLayout2.addLayout(formLayout)
        hLayout2.addLayout(vLayout)
        vLayout2.addLayout(hLayout1)
        vLayout2.addLayout(hLayout2)

        # now that widgets are created, populate with saved values
        load_settings(self)

        self.update_widgets()

        self.show()
        self.raise_()

    def reimport_libs(self):
        print('Re-importing GUI libraries')
        importlib.reload(browser)
        importlib.reload(editor)

    def resizeEvent(self, event):
        save_settings(self)

    def moveEvent(self, event):
        save_settings(self)

    def update_widgets(self):

        batch = int(self.batchLE.text())
        cellmask = self.cellLE.text() + "%"
        # strip white spaces and split by semicolon
        modeltext = self.modelLE.text().replace(' ', '').split(';')
        # attach % wildcard before and after modelstring
        modelmask = ['%' + m + '%' for m in modeltext]
        #modelmask = "%" + self.modelLE.text() + "%"

        save_settings(self)

        if batch > 0:
            self.batch = batch
        else:
            self.batchLE.setText(str(self.batch))
        #" ORDER BY cellid",

        #self.d_cells = nd.get_batch_cells(self.batch, cellid=cellmask)

        self.d_cells = nd.pd_query("SELECT DISTINCT cellid FROM NarfResults" +
                               " WHERE batch=%s AND cellid like %s" +
                               " ORDER BY cellid",
                               (self.batch, cellmask))

        modelquery = ("SELECT modelname, count(*) as n, max(lastmod) as "
                      "last_mod FROM NarfResults WHERE batch=%s AND ")
        for i, m in enumerate(modelmask):
            modelquery += 'modelname like %s OR '
        modelquery = modelquery[:-3]  # drop the trailing OR
        modelquery += 'GROUP BY modelname ORDER BY modelname'
        self.d_models = nd.pd_query(modelquery, (self.batch, *modelmask))

#        self.d_models = nd.pd_query("SELECT modelname, count(*) as n, max(lastmod) as last_mod FROM NarfResults" +
#                               " WHERE batch=%s AND modelname like %s" +
#                               " GROUP BY modelname ORDER BY modelname",
#                               (self.batch, modelmask))

        self.cells.clear()
        for c in list(self.d_cells['cellid']):
            list_item = qw.QListWidgetItem(c, self.cells)

        self.models.clear()
        for m in list(self.d_models['modelname']):
            list_item = qw.QListWidgetItem(m, self.models)
        #self.data_model._df = self.d_models
        #self.data_table.setModel(self.data_model)

        print('updated list widgets')

    def _selected_modelnames(self):
        items = [a.text() for a in self.models.selectedItems()]
        return items

    def _selected_cells(self):
        items = [a.text() for a in self.cells.selectedItems()]
        return items

    def run_fun(self):
        cellids = self._selected_cells()
        modelnames = self._selected_modelnames()
        batch = self.batch

        d = nd.batch_comp(batch=batch, modelnames=modelnames, cellids=cellids, stat='r_test')

        cellids = d.index
        siteids = [c.split("-")[0] for c in cellids]

        d['siteids'] = siteids
        mean_r_test = d[modelnames].mean().values

        shortened, prefix, suffix = find_common(modelnames)
        modelcount=len(modelnames)
        plt.figure()
        ax = plt.subplot(1,1,1)
        site_r_test = d.groupby(['siteids']).mean().values.T

        usiteids=d.groupby(['siteids']).groups.keys()

        ax.bar(np.arange(len(mean_r_test)), mean_r_test, color='lightgray')
        ax.plot(site_r_test)
        print('{} -- {}'.format(prefix, suffix))
        for i,m in enumerate(modelnames):

            print("{} (n={}): {:.3f}".format(shortened[i], d[m].count(), d[m].mean()))
            r = d[m].values
            #plt.plot(np.random.uniform(low=-0.25, high=0.25, size=r.shape)+i,
            #         r, '.', color='gray')
            s = shortened[i].replace("_","\n") + "\n{:.3f}".format(mean_r_test[i])
            ax.text(i, 0, s, rotation=90, color='black',
                    ha='left', va='bottom', fontsize=7)

        for i, s in enumerate(usiteids):
            ax.text(modelcount-0.2, site_r_test[-1,i], s)

        plt.title('{} -- {}'.format(prefix, suffix))
        ax.set_xticks(np.arange(modelcount))
        ax.set_xticklabels([])

        nplt.ax_remove_box(ax)

    def get_current_selection(self):
        aw = self

        batch = aw.batch
        try:
            cellid = aw.cells.currentItem().text()
            modelname = aw.models.currentItem().text()
            #print(cellid)
            #print("Model(s) selected")
            #for index in range(aw.models.count()):
                #print(aw.models.item(index).text())
        except AttributeError:
            print("You must select a cellid and modelname first")
            return None, None

        print("Viewing {},{},{}".format(batch,cellid,modelname))

        if (aw.last_loaded[0]==cellid and aw.last_loaded[1]==modelname and
            aw.last_loaded[2]==batch):
            xf = aw.last_loaded[3]
            ctx = aw.last_loaded[4]
        else:
            xf, ctx = xhelp.load_model_xform(cellid, batch, modelname, eval_model=True)
            aw.last_loaded=[cellid,modelname,batch,xf,ctx]
        return xf, ctx

    def view_recording(self):
        aw = self

        batch = aw.batch
        cellid = aw.cells.currentItem().text()
        modelname = aw.models.currentItem().text()
        xf, ctx = aw.get_current_selection()
        if xf is None:
            return

        recname=aw.recname
        #signals = ['stim','psth','state','resp','pred','mask']
        signals = ['stim','psth','state','resp','pred']
        if type(ctx[recname]) is list:
            rec = ctx[recname][0].apply_mask()
            #rec = ctx[recname][0]
        else:
            rec = ctx[recname].copy()

        aw2 = browser.browse_recording(rec, signals=signals,
                               cellid=cellid, modelname=modelname)

        self._cached_windows.append(aw2)
        return aw2

    def view_model(self):
        aw = self
        xf, ctx = aw.get_current_selection()
        if xf is None:
            return

        #browse_xform_fit(ctx, xf)
        self.ex = editor.EditorWidget(modelspec=ctx['modelspec'], rec=ctx['val'], xfspec=xf,
                                 ctx=ctx, parent=self)
        self.ex.show()
        #nplt.quickplot(ctx)

    def compare_models(self):
        models = self._selected_modelnames()
        cells = self._selected_cells()
        batch = self.batch

        self.cx = comparison.ComparisonWidget(batch, cells, models, parent=self)
        self.cx.show()


def view_model_recording(cellid="TAR010c-18-2", batch=289,
                         modelname="ozgf.fs100.ch18-ld-sev_dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1_init-basic",
                         recname='val'):

    xf, ctx = nw.load_model_baphy_xform(cellid, batch, modelname, eval_model=True)

    #signals = ['stim','psth','state','resp','pred']
    signals = ['stim','psth','state','resp','pred','mask']
    #rec = ctx[recname][0].apply_mask()
    rec = ctx[recname][0]
    aw = browser.browse_recording(rec, signals=signals,
                                  cellid=cellid, modelname=modelname)
    return aw


if __name__ == '__main__':
    print(sys.argv[0])
    if sys.argv[0] != '':
        app = qw.QApplication(sys.argv)
        m = model_browser()
        sys.exit(app.exec_())

