### Test runner.py
import platform
import sys
import logging

logging.info(f"Platform: {sys.platform}")

logging.info(f"Machine: {platform.machine()}")

logging.info(f"Release: {platform.release()}")