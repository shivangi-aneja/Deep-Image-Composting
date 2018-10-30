"""
class file for logging
"""
import logging
from datetime import datetime


# Logging utilities - logs get saved in folder logs named by date and time, and also output
# at standard output

logFormatter = logging.Formatter("[%(asctime)s]  %(message)s", datefmt='%m/%d %I:%M:%S')

rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler(datetime.now().strftime('logs/log_%d_%m_%H_%S.log'))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
