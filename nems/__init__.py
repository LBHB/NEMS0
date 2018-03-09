import logging
log = logging.getLogger(__name__)


def load_config():
    # Load the default settings
    from os import environ
    from . import default_config as config

    try:
        # Load the computer-specific settings
        path = environ['NEMS_CONFIG']
        import imp
        from os.path import dirname
        extra_settings = imp.load_module('settings', open(path), dirname(path),
                                         ('.py', 'r', imp.PY_SOURCE))
        # Update the setting defaults with the computer-specific settings
        for setting in dir(extra_settings):
            value = getattr(extra_settings, setting)
            setattr(config, setting, value)
        config.init_settings()
    except KeyError:
        # We repeat config.init_settings before each logging message n the
        # except block (to ensure that logging gets configured properly so the
        # message appears in the right places).
        config.init_settings()
        log.warn('No NEMS_CONFIG defined')
    except IOError:
        config.init_settings()
        log.warn('%s file defined by NEMS_CONFIG is missing', path)

    return config


_config = load_config()


def get_config(setting):
    '''
    Get value of setting.
    '''
    return getattr(_config, setting)
