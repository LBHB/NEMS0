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

import PyQt5.QtCore as qc
import PyQt5.QtGui as qg
import PyQt5.QtWidgets as qw

import nems_db.xform_wrappers as nw
import nems.plots.api as nplt
import nems.xforms as xforms
import nems.epoch as ep
from nems.utils import find_module
import pandas as pd
import scipy.ndimage.filters as sf
import nems_lbhb.plots as lplt
import nems.db as nd
from nems.gui.recording_browser import (browse_recording)
from nems.gui.editors import EditorWidget

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

        hLayout = qw.QHBoxLayout(self)

        self.cells = qw.QListWidget(self)
        self.cells.setMaximumSize(qc.QSize(130, 1000))
        self.models = qw.QListWidget(self)

        vLayout = qw.QVBoxLayout(self)

        formLayout = qw.QFormLayout(self)
        batchlabel = qw.QLabel(self)
        batchlabel.setText('Batch:')
        self.batchLE = qw.QLineEdit(self)
        self.batchLE.setText(str(batch))
        self.batchLE.returnPressed.connect(self.update_widgets)
        self.batchLE.setMaximumWidth(800)
        formLayout.addRow(batchlabel, self.batchLE)

        celllabel = qw.QLabel(self)
        celllabel.setText('CellID:')
        self.cellLE = qw.QLineEdit(self)
        self.cellLE.setText(cell_search_string)
        self.cellLE.returnPressed.connect(self.update_widgets)
        self.cellLE.setMaximumWidth(800)
        formLayout.addRow(celllabel,self.cellLE)

        modellabel = qw.QLabel(self)
        modellabel.setText('Modelname:')
        self.modelLE = qw.QLineEdit(self)
        self.modelLE.setText(model_search_string)
        self.modelLE.returnPressed.connect(self.update_widgets)
        self.modelLE.setMaximumWidth(800)

        formLayout.addRow(modellabel,self.modelLE)

        vLayout.addLayout(formLayout)

        self.updateBtn = qw.QPushButton("Update lists", self)
        self.updateBtn.clicked.connect(self.update_widgets)
        vLayout.addWidget(self.updateBtn)

        self.viewBtn = qw.QPushButton("View recording", self)
        self.viewBtn.clicked.connect(self.view_recording)
        vLayout.addWidget(self.viewBtn)

        self.modelBtn = qw.QPushButton("View model", self)
        self.modelBtn.clicked.connect(self.view_model)
        vLayout.addWidget(self.modelBtn)

        hLayout.addLayout(vLayout)
        hLayout.addWidget(self.cells)
        hLayout.addWidget(self.models)

        self.update_widgets()

        self.show()
        self.raise_()

    def update_widgets(self):

        batch = int(self.batchLE.text())
        cellmask = self.cellLE.text()
        modelmask = "%" + self.modelLE.text() + "%"

        if batch > 0:
            self.batch = batch
        else:
            self.batchLE.setText(str(self.batch))

        self.d_cells = nd.get_batch_cells(self.batch, cellid=cellmask)
        self.d_models = nd.pd_query("SELECT DISTINCT modelname FROM NarfResults" +
                               " WHERE batch=%s AND modelname like %s" +
                               " ORDER BY modelname",
                               (self.batch, modelmask))

        self.cells.clear()
        for c in list(self.d_cells['cellid']):
            list_item = qw.QListWidgetItem(c, self.cells)

        self.models.clear()
        for m in list(self.d_models['modelname']):
            list_item = qw.QListWidgetItem(m, self.models)

        print('updated list widgets')

    def get_current_selection(self):
        aw = self

        batch = aw.batch
        cellid = aw.cells.currentItem().text()
        modelname = aw.models.currentItem().text()

        print("Viewing {},{},{}".format(batch,cellid,modelname))

        if (aw.last_loaded[0]==cellid and aw.last_loaded[1]==modelname and
            aw.last_loaded[2]==batch):
            xf = aw.last_loaded[3]
            ctx = aw.last_loaded[4]
        else:
            xf, ctx = nw.load_model_baphy_xform(cellid, batch, modelname, eval_model=True)
            aw.last_loaded=[cellid,modelname,batch,xf,ctx]
        return xf, ctx

    def view_recording(self):
        aw = self

        batch = aw.batch
        cellid = aw.cells.currentItem().text()
        modelname = aw.models.currentItem().text()
        xf, ctx = aw.get_current_selection()

        recname=aw.recname
        #signals = ['stim','psth','state','resp','pred','mask']
        signals = ['stim','psth','state','resp','pred']
        if type(ctx[recname]) is list:
            rec = ctx[recname][0].apply_mask()
            #rec = ctx[recname][0]
        else:
            rec = ctx[recname].copy()

        aw2 = browse_recording(rec, signals=signals,
                               cellid=cellid, modelname=modelname)

        self._cached_windows.append(aw2)
        return aw2

    def view_model(self):
        aw = self
        xf, ctx = aw.get_current_selection()
        ex = EditorWidget(modelspec=ctx['modelspec'], rec=ctx['val'], xfspec=xf,
                          ctx=ctx, parent=self)
        ex.show()
        #nplt.quickplot(ctx)


def view_model_recording(cellid="TAR010c-18-2", batch=289,
                         modelname="ozgf.fs100.ch18-ld-sev_dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1_init-basic",
                         recname='val'):

    xf, ctx = nw.load_model_baphy_xform(cellid, batch, modelname, eval_model=True)

    #signals = ['stim','psth','state','resp','pred']
    signals = ['stim','psth','state','resp','pred','mask']
    #rec = ctx[recname][0].apply_mask()
    rec = ctx[recname][0]
    aw = browse_recording(rec, signals=signals,
                         cellid=cellid, modelname=modelname)
    return aw

if __name__ == '__main__':
    app = qw.QApplication(sys.argv)
    m = model_browser()
    #ex = EditorWindow(modelspec=modelspec, xfspec=xfspec, rec=rec, ctx=ctx)
    sys.exit(app.exec_())

