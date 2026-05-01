import logging
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            format="%(asctime)s%(levelname)s:%(message)s",
            stream=sys.stdout,
            level=logging.INFO,
        )
        _CONFIGURED = True
    return logging.getLogger(name)
