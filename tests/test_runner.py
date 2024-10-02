"""Tests for the runner environment."""

from __future__ import annotations

import logging
import platform
import sys

logging.info(f"Platform: {sys.platform}")

logging.info(f"Machine: {platform.machine()}")

logging.info(f"Release: {platform.release()}")
