from __future__ import annotations

import logging
import signal
from typing import Any

logger = logging.getLogger("mqtpredictor")


def timeout_watcher(func: Any, args: list[Any], timeout: int) -> Any:
    """Method that stops a function call after a given timeout limit."""

    class TimeoutException(Exception):  # Custom exception class
        pass

    def timeout_handler(_signum: Any, _frame: Any) -> None:  # Custom signal handler
        raise TimeoutException

    # Change the behavior of SIGALRM
    signal.signal(signal.SIGALRM, timeout_handler)

    signal.alarm(timeout)
    try:
        res = func(*args)
    except TimeoutException:
        logger.debug("Calculation/Generation exceeded timeout limit for " + func.__module__ + ", " + str(args[1:]))
        return False
    except Exception as e:
        logger.error("Something else went wrong: " + str(e))
        return False
    else:
        # Reset the alarm
        signal.alarm(0)

    return res
