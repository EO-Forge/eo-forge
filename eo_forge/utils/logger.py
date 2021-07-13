import logging
import logging.config

# TODO: Migrate to logging config file or dict


def update_logger(logger, msg, log_type):
    """
    :param msg: message to be added
    :param type: one of logging.levels
    """
    levels = {
        "CRITICAL": 50,
        "ERROR": 40,
        "WARNING": 30,
        "INFO": 20,
        "DEBUG": 10,
    }

    assert log_type in list(levels)

    if logger is None:
        print("Logger NOT SET - trying to log: {}-{}".format(log_type, msg))
    else:
        logger.log(levels[log_type], msg)


def log_fmt():
    return {
        "notime": "%(name)s - %(levelname)s - %(message)s",
        "time_detailed": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }


class basic_logger(object):
    """"""

    def __init__(self, logger_name, main_logger_level=logging.DEBUG):
        """
        :param logger_name: logger name to be used
        :param main_logger_level: main logger logging level
        """
        self.logger_ = logging.getLogger(logger_name)
        self.logger_.setLevel(main_logger_level)
        self.logger_main_level_ = main_logger_level
        self.handlers_ = []

    def set_handler(
        self,
        handler,
        log_level=logging.INFO,
        log_fmt=log_fmt()["time_detailed"],
    ):
        """
        :param handler: logging.StremHandler(),logging.FileHandler('file2log'),
        :param log_level: logging.INFO,logging.ERROR,etc
        :param log_fmt: logging.Formatter
        """
        #
        t_handler = handler
        t_handler.setLevel(log_level)
        if log_level < self.logger_main_level_:
            print("Handler Logging Level lower Than Main Logger")
        t_format = logging.Formatter(log_fmt)
        t_handler.setFormatter(t_format)
        self.handlers_.append(t_handler)
        t_handler = None
        t_format = None

    def add_handlers(self):
        """
        :param none
        """
        for h in self.handlers_:
            self.logger_.addHandler(h)
