import json
import logging

class Logger:
    def __init__(self, filename):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        file_handler = logging.FileHandler(filename, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%d %b %Y %H:%M:%S')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        logging.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def info(self, strs):
        return self.logger.info(strs)

    def warning(self, strs):
        return self.logger.warning(strs)

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)