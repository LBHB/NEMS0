import logging
import sys
from pathlib import Path
from configparser import ConfigParser, DuplicateSectionError

from PyQt5 import uic
from PyQt5.QtWidgets import *

from nems0 import get_setting
from nems.gui_new.db_browser.ui_promoted import ListViewListModel

# TEMP ERROR CATCHER
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook

import matplotlib.pyplot as plt
font_size = 8
params = {'legend.fontsize': font_size,
          'figure.figsize': (8, 6),
          'axes.labelsize': font_size,
          'axes.titlesize': font_size,
          'axes.spines.right': False,
          'axes.spines.top': False,
          'xtick.labelsize': font_size,
          'ytick.labelsize': font_size,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
plt.rcParams.update(params)


def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


# Set the exception hook to our wrapping function
sys.excepthook = my_exception_hook

log = logging.getLogger(__name__)

# read and parse the UI file to generate the GUI
qt_creator_file = Path(r'ui') / 'container.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(qt_creator_file)


class MainWindow(QtBaseClass, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # need to init these before initing child models
        self.db_conns = []  # track connections to gracefully close
        self.load_saved_settings()

        # setup relationships
        self.tabBrowser.parent = self
        self.tabComp.parent = self
        self.tabAnalysis.parent = self
        self.children = [self.tabBrowser, self.tabComp, self.tabAnalysis]

        # init top level models (those shared by multiple tabs)
        self.init_models()

        # init models in children
        for child in self.children:
            print(child)
            child.init_models()

        # setup the callbacks for the menu items
        self.actionOpen.triggered.connect(self.tabBrowser.on_action_open)
        self.actionCustom_function.triggered.connect(self.tabBrowser.on_action_custom_function)

        self.actionSave_selections.triggered.connect(self.on_action_save_selections)
        self.actionLoad_selections.triggered.connect(self.on_action_load_selections)

        # set the filter to be in focus
        if self.tabWidget.currentWidget().tab_name == 'browser_tab':
            self.tabBrowser.lineEditModelFilter.setFocus()
        elif self.tabWidget.currentWidget().tab_name == 'comp_tab':
            self.tabComp.lineEditCellFilter.setFocus()

    def closeEvent(self, event):
        """Catch close event in order to close db connections and save selections."""
        for db_conn in self.db_conns:
            db_conn.close_db()

        self.save_settings()
        plt.close('all')
        event.accept()

    def init_models(self):
        """Init models shared among the tabs."""
        self.batchModel = ListViewListModel(table_name='Results', column_name='batch')
        self.db_conns.append(self.batchModel)

    def load_saved_settings(self):
        """Reads in saved settings.

        The section header for this GUI is "db_browser", with each tab saving it's config
        in variables prefixed with their tab name. Different configs are named by appending
        a colon then the save name (ex: "db_browser:save1").
        """
        self.config_file = Path(get_setting('SAVED_SETTINGS_PATH')) / 'gui.ini'
        self.config_group = 'db_browser'
        self.config = ConfigParser(delimiters=('='))
        self.config.read(self.config_file)

    def save_settings(self, group_name=None):
        """Collects selections from tabs and saves them."""
        config_group = self.config_group
        if group_name is not None:
            config_group = config_group + ':' + group_name

        to_save = {}
        for child in self.children:
            selections = child.get_selections()
            # check no duplicate keys
            assert not set(to_save) & set(selections), 'Overlapping save parameters between tabs.'
            to_save.update(selections)

        # add section if not present
        try:
            self.config.add_section(config_group)
        except DuplicateSectionError:
            pass

        for k, v in to_save.items():
            self.config.set(config_group, k, v)

        with open(self.config_file, 'w') as cf:
            self.config.write(cf)

    def get_saved_sections(self):
        """Gets a list of the sections in the settings."""
        return [section.lstrip('db_browser:') for section in self.config.sections() if section.startswith(
            'db_browser:')]

    def on_action_save_selections(self, *args, text=''):
        """Event handler for saving selections."""
        input_text, accepted = QInputDialog.getText(self, 'Selection name', 'Enter selection name:', text=text)
        if not accepted or not input_text:
            return

        existing = self.get_saved_sections()

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
        existing = self.get_saved_sections()
        if not existing:
            ret = QMessageBox.warning(self, 'Load', 'No saved selections to load!', QMessageBox.Close)
            return

        group_name, accepted = QInputDialog.getItem(self, 'Selection name', 'Select settings to load:', existing,
                                                    editable=False)
        if not accepted:
            return

        status_text = f'Loading saved selections: "{group_name}"...'
        log.info(status_text)
        self.statusbar.showMessage(status_text)

        for child in self.children:
            child.update_selections(*child.load_settings(group_name=group_name))

        self.statusbar.clearMessage()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
