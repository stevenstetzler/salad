import pickle
import sys
import logging
from pathlib import Path

logging.basicConfig()
log = logging.getLogger(__name__)

class Serializable():
    @classmethod
    def read(cls, file):
        if file is sys.stdin:
            file = sys.stdin.buffer

        log.debug("reading from %s", file)
        if isinstance(file, str) or isinstance(file, Path):
            with open(file, "rb") as fd:
                return pickle.load(fd)
        else:
            return pickle.load(file)
    
    def write(self, file):
        if file is sys.stdout:
            file = sys.stdout.buffer

        log.debug("writing to %s", file)
        if isinstance(file, str) or isinstance(file, Path):
            with open(file, "wb") as fd:
                pickle.dump(self, fd)
        else:
            pickle.dump(self, file)

def read(file):
    return Serializable.read(file)

def write(object, file):
    return Serializable.write(object, file)

