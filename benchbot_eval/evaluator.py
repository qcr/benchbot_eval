import pickle

from .validator import DUMP_LOCATION, SKIP_KEY


class Evaluator:
    def __init__(self, skip_load=False):

        if not skip_load:
            self.load_validator()

    def load_validator(self):
        with open(DUMP_LOCATION, 'rb') as f:
            self.__dict__.update(pickle.load(f).__dict__)

    def evaluate(self):
        print(self.__dict__)
