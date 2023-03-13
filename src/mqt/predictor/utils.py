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


NUM_QUBIT_INDICES_RIGETTI = 80


def get_rigetti_qubit_dict() -> dict[str, str]:
    mapping = {
        "32": "4",
        "39": "3",
        "38": "2",
        "37": "1",
        "36": "0",
        "35": "7",
        "34": "6",
        "33": "5",
        "25": "15",
        "24": "14",
        "31": "13",
        "30": "12",
        "29": "11",
        "28": "10",
        "27": "17",
        "26": "16",
        "17": "25",
        "16": "24",
        "23": "23",
        "22": "22",
        "21": "21",
        "20": "20",
        "19": "27",
        "18": "26",
        "8": "34",
        "9": "35",
        "10": "36",
        "11": "37",
        "12": "30",
        "13": "31",
        "14": "32",
        "15": "33",
        "0": "44",
        "1": "45",
        "2": "46",
        "3": "47",
        "4": "40",
        "5": "41",
        "6": "42",
        "7": "43",
        "72": "104",
        "73": "105",
        "74": "106",
        "75": "107",
        "76": "100",
        "77": "101",
        "78": "102",
        "79": "103",
        "64": "114",
        "65": "115",
        "66": "116",
        "67": "117",
        "68": "110",
        "69": "111",
        "70": "112",
        "71": "113",
        "56": "124",
        "57": "125",
        "58": "126",
        "59": "127",
        "60": "120",
        "61": "121",
        "62": "122",
        "63": "123",
        "48": "134",
        "49": "135",
        "50": "136",
        "51": "137",
        "52": "130",
        "53": "131",
        "54": "132",
        "55": "133",
        "40": "144",
        "41": "145",
        "42": "146",
        "43": "147",
        "44": "140",
        "45": "141",
        "46": "142",
        "47": "143",
    }

    assert len(mapping) == NUM_QUBIT_INDICES_RIGETTI
    return mapping
