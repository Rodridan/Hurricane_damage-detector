# logging_setup.py
from loguru import logger
import os

def setup_logging():
    os.makedirs("outputs", exist_ok=True)
    logger.add("outputs/hurricane_detector.log", rotation="1 MB")
