'''
Configuration file for NEMS containing reasonable defaults

Variable names in capital letters indicate this is a setting that can be
overriden by a custom file that the NEMS_CONFIG environment variable points to.
Use virtual environments (via virtualenv or conda) for creating separate
installations of NEMS that use different configuration files.
'''
# Note for developers. See `__init__.py` to understand how this integrates with
# the file specified by NEMS_CONFIG.
import logging.config
import socket
import datetime
import os.path
import os


################################################################################
# System information
################################################################################
# Name of computer
SYSTEM = socket.gethostname()


################################################################################
# Logging configuration
################################################################################
# Folder to store log file in
NEMS_LOG_ROOT = '/tmp/nems'

# Filename to save log file in
NEMS_LOG_FILENAME = datetime.datetime.now().strftime('NEMS %Y-%m-%d %H%M%S.log')

# Format for messages saved to file
NEMS_LOG_FILE_FORMAT = '[%(relativeCreated)d %(thread)d %(name)s - %(levelname)s] %(message)s'

# Format for messages printed to console
NEMS_LOG_CONSOLE_FORMAT = '[%(name)s %(levelname)s] %(message)s'

# Logging level for file
NEMS_LOG_FILE_LEVEL = 'DEBUG'

# Logging level for console
NEMS_LOG_CONSOLE_LEVEL = 'DEBUG'


################################################################################
# Data & database
################################################################################

NEMS_DIR = os.path.abspath(os.path.dirname(__file__) + '/../../')
NEMS_RESULTS_DIR = os.path.abspath(os.path.dirname(__file__) + '/../../results')
NEMS_RECORDINGS_DIR = os.path.abspath(os.path.dirname(__file__) + '/../../recordings')
SAVED_SETTINGS_PATH = os.path.abspath(os.path.dirname(__file__))

SQL_ENGINE = 'sqlite'

MYSQL_HOST = None
MYSQL_USER = None
MYSQL_PASS = None
MYSQL_DB = None
MYSQL_PORT ='3306'

NEMS_BAPHY_API_PORT = '3003'
NEMS_BAPHY_API_HOST = 'neuralprediction.org'
USE_NEMS_BAPHY_API = False

# Default paths passed to command prompt for model queue
# TO DO -- FIX PATHS TO SOMETHING GENERIC
DEFAULT_EXEC_PATH = '/auto/users/nems/anaconda3/bin/python'
DEFAULT_SCRIPT_PATH = '/auto/users/nems/nems_db/nems_fit_single.py'

# (cmd line function called every queue tick--to update queue load in tComputers)
QUEUE_TICK_EXTERNAL_CMD = '/auto/users/svd/python/nems_db/bin/qsetload'

################################################################################
# Plugins Registries
################################################################################
# Keyword Plugins, updates nems0.plugins.default_keywords
# ex: KEYWORD_PLUGINS = ['/path/to/keywords/', '/second/path/']
KEYWORD_PLUGINS = []

# Xforms Plugins, updates nems0.plugins.default_loaders, default_fitters, etc.
# ex: XF_LOADER_PLUGINS = ['/path/to/my/plugins/loaders/', '/second/path/']
XFORMS_PLUGINS = []

# alternative keyword system where libraries are imported and pulled in as 
# keywords according to decorators
LIB_PLUGINS = []

################################################################################
# Display tweaks
################################################################################
# Name of computer
FILTER_CMAP = 'jet'   # alternative is 'RdYlBu_r'
FILTER_INTERPOLATION = None
WEIGHTS_CMAP = 'bwr'


################################################################################
# Post config
################################################################################
def configure_logging(filename=None):
    # Actual configuration needs to be done inside this function (called by
    # init_settings) so that values in the NEMS_CONFIG file can override these.
    config = {
        'version': 1,
        'formatters': {
            'file': {'format': NEMS_LOG_FILE_FORMAT},
            'console': {'format': NEMS_LOG_CONSOLE_FORMAT},
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'console',
                'level': NEMS_LOG_CONSOLE_LEVEL,
            },
        },
        'loggers': {
            '__main__': {'level': 'INFO'},
            '': {'level': 'INFO'},
            'nems_db': {'level': 'INFO'},
            'nems': {'level': 'INFO'},
            'nems0.analysis.fit_basic': {'level': 'INFO'},
            'fontTools': {'level': 'WARNING'}
        },
        'root': {
            'handlers': ['console'],
        },
    }

    if filename is not None:
        file_config = {
            'class': 'logging.FileHandler',
            'formatter': 'file',
            'filename': filename,
            'mode': 'w',
            'encoding': 'UTF-8',
            'level': NEMS_LOG_FILE_LEVEL,
        }
        config['handlers']['file'] = file_config
        config['root']['handlers'].append('file')

    logging.config.dictConfig(config)


def init_settings():
    log = logging.getLogger(__name__)

    # This code is called by __init__.py after extra configuration files
    # (specified by NEMS_CONFIG) are loaded. This ensures that all variables are
    # set properly.
    if NEMS_LOG_FILENAME is not None:
        log_filename = os.path.join(NEMS_LOG_ROOT, NEMS_LOG_FILENAME)
        os.makedirs(NEMS_LOG_ROOT, exist_ok=True)
        # make LOG directory world-writeable
        try:
            os.chmod(NEMS_LOG_ROOT, 0o777)
        except:
            pass
        configure_logging(log_filename)
        log.info("Saving log messages to %s", log_filename)
    else:
        configure_logging()

    # Log the settings to facilitate debugging. By convention, settings are in
    # all caps, so only log those variables.
    log = logging.getLogger(__name__)
    for k, v in sorted(globals().items()):
        if k == k.upper():
            log.debug("CONFIG %s : %r", k, v)
