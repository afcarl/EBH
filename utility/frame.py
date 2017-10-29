import os
import gzip
import pickle

from .parse import extract_data
from EBH import logroot


class DataWrapper:

    def __init__(self, source):
        if ".txt" == source[-4:]:
            self.data = extract_data(source)
            self.ID = os.path.split(source)[-1].split(".")[0]
        elif ".wrp" == source[-4:]:
            self.ID, self.data = DataWrapper.load(source)
        else:
            # Assume source is ID
            self.ID = source
            self.data = extract_data(logroot + f"{source}.txt")

    @property
    def left(self):
        return self.data[:2]

    @property
    def right(self):
        return self.data[2:]

    @staticmethod
    def load(source):
        return pickle.load(gzip.open(source))

    def save(self, dest):
        with gzip.open(dest) as handle:
            pickle.dump(self, handle)
