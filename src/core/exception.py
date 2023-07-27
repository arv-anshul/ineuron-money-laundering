""" Custom Exception for the project. Shows the filename and line number. """

import logging
from sys import exc_info
from types import TracebackType
from typing import TypeAlias

ExcInfo: TypeAlias = tuple[type[BaseException], BaseException, TracebackType]
OptExcInfo: TypeAlias = ExcInfo | tuple[None, None, None]


class CustomException(Exception):
    def __init__(
        self, message: object, detail: OptExcInfo, logger: logging.Logger,
    ) -> None:
        super().__init__(message)
        self.logger = logger
        self.message = self.construct_message(message, detail)

    def construct_message(
        self, error: object, error_detail: OptExcInfo,
    ) -> str:
        _, _, exc_tb = error_detail
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            message = f'"{file_name}":[{exc_tb.tb_lineno}] - {error}'
        else:
            message = 'No error details available for the raised exception.'
        self.logger.exception(message)
        return message

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return CustomException.__name__

    @classmethod
    def wrap_with_custom_exception(cls, logger: logging.Logger):
        """
        Wraps all methods of a class in a try-except block and raises a custom exception.
        """
        def wrapper(method):
            def wrapped(*args, **kwargs):
                try:
                    return method(*args, **kwargs)
                except Exception as e:
                    raise CustomException(
                        e, exc_info(), logger,
                    ) from e
            return wrapped

        def decorator(cls):
            for name, method in vars(cls).items():
                if callable(method):
                    setattr(cls, name, wrapper(method))
            return cls

        return decorator
