from typing import Optional

import logging
import os
from pathlib import Path


class Logger:
    logger = {}

    def __init__(self, log_file: Optional[Path] = None):

        logger_name = self.__class__.logger_name

        if logger_name not in self.__class__.logger:
            logger = logging.getLogger(logger_name)
            
            formatter = logging.Formatter("%(message)s")
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            if log_file:
                log_file.parent.mkdir(parents=True, exist_ok=True)

                formatter = logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s"
                )

                handler = logging.FileHandler(log_file)
                handler.setFormatter(formatter)
                logger.addHandler(handler)

            level = os.environ.get("CENAI_LOG_LEVEL", "info")
            logger.setLevel(logging.__dict__[level.upper()])

            self.__class__.logger[logger_name] = logger

    @classmethod
    def tweak_logger(cls) -> None:
        logger = cls.logger[cls.logger_name]

        for handler in logger.handlers:
            logger.removeHandler(handler)

    @classmethod
    def set_level(cls, level: str) -> str:
        logger = cls.logger[cls.logger_name]

        current = logging.getLevelName(logger.level).lower()
        logger.setLevel(logging.__dict__[level.upper()])
        return current

    @classmethod
    def is_enebled_for(cls, level: str) -> bool:
        logger = cls.logger[cls.logger_name]

        return logger.isEnabledFor(logging.__dict__[level.upper()])

    @classmethod
    def LOG(cls, level: str, mesg: str, *args, **kwargs) -> None:
        logger = cls.logger[cls.logger_name]

        logger.log(
            logging.__dict__[level.upper()],
            mesg, *args, stacklevel=2, **kwargs
        )

    @classmethod
    def DEBUG(cls, mesg: str, *args, **kwargs) -> None:
        logger = cls.logger[cls.logger_name]
        logger.debug(mesg, *args, stacklevel=2, **kwargs)

    @classmethod
    def INFO(cls, mesg: str, *args, **kwargs) -> None:
        logger = cls.logger[cls.logger_name]
        logger.info(mesg, *args, stacklevel=2, **kwargs)

    @classmethod
    def WARNING(cls, mesg: str, *args, **kwargs) -> None:
        logger = cls.logger[cls.logger_name]
        logger.warning(mesg, *args, stacklevel=2, **kwargs)

    @classmethod
    def ERROR(cls, mesg: str, *args, **kwargs) -> None:
        logger = cls.logger[cls.logger_name]
        logger.error(mesg, *args, stacklevel=2, **kwargs)

    @classmethod
    def CRITICAL(cls, mesg: str, *args, **kwargs) -> None:
        logger = cls.logger[cls.logger_name]
        logger.critical(mesg, *args, stacklevel=2, **kwargs)
