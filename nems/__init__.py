import logging
log = logging.getLogger(__name__)


def load_config():
    # Load the default settings
    from os import environ, path, utime
    from configs import defaults as config
    # leave defaults.py off the end
    configs_path = path.abspath(config.__file__)[:-11]

    try:
        from configs import settings
    except ImportError:
        settings_path = path.join(configs_path, 'settings.py')
        # this should be equivalent to
        # `touch path/to/configs/settings.py`
        with open(settings_path, 'a'):
            utime(settings_path, None)
        log.info("No settings.py found in configs directory,"
                 " generating blank file ... ")
        from configs import settings

    for s in config.__dir__():
        if s.startswith('__') or not (s == s.upper()):
            # Ignore python magic variables. Everything else in
            # the defaults files should be valid settings.
            pass
        elif s == s.upper():
            if s in environ:
                # If it's already in the environment, don't need
                # to do anything else.
                pass
            elif hasattr(settings, s):
                log.info("Found setting: %s in %s, adding "
                         "value to environment ... ", s, settings.__name__)
                d = getattr(settings, s)
                if d is None:
                    d = ''
                environ[s] = d
            else:
                log.info("No value specified for: %s. Using default value "
                         "in %s", s, config.__name__)
                d = getattr(config, s)
                if d is None:
                    d = ''
                environ[s] = d
            setattr(config, s, environ[s])

    config.init_settings()
    return config


_config = load_config()


def get_setting(setting):
    '''
    Get value of setting.
    '''
    return getattr(_config, setting)
