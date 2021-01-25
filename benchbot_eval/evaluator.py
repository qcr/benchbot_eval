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
        # Ensure we have some valid results
        valid_results = [
            k for k, v in self.results_data.items() if not v[SKIP_KEY]
        ]
        if not valid_results:
            print("Exiting, as no valid results were provided.")

        print("Running: %s" % valid_results)
