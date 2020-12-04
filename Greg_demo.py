#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:58:16 2020

@author: luke
"""
import SPO_helpers as sp

#Decide which cell:
#UPDATE THE LINE BELOW TO POINT TO THE FILE
rec_file_dir='/Users/grego/Downloads/'
#cellid='fre196b-21-1_1417-1233'; rf='fre196b_ec1d319ae74a2d790f3cbda73d46937e588bc791.tgz'
cellid='fre197c-105-1_705-1024'; rf='fre197c_f94fb643b4cb6380b8eb3286fc30d908a1940ea2.tgz' #Neuron 1 on poster
batch=306
rec_file = rec_file_dir + rf


# To plot PSTHs if you have acess to the database:
sp.plot_linear_and_weighted_psths(batch,cellid)

# To plot PSTHs with a local file
sp.plot_linear_and_weighted_psths(batch,cellid,rec_file=rec_file)

# There's a bunch of stuff in here manipulating the PSTHs to calculate various metrics on them, might be useful.
#As above, if you have the database:
sp.calc_psth_metrics(batch,cellid)

#Or with a local file:
sp.calc_psth_metrics(batch,cellid,rec_file)
