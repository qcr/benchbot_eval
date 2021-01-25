import pickle

from . import helpers


class Evaluator:
    def __init__(self, evaluation_methods_filenames, skip_load=False):

        if not skip_load:
            self.load_validator()

    def load_validator(self):
        with open(helpers.DUMP_LOCATION, 'rb') as f:
            self.__dict__.update(pickle.load(f).__dict__)

    def evaluate(self):
        # Ensure we have some valid results
        valid_results = [
            k for k, v in self.results_data.items() if not v[helpers.SKIP_KEY]
        ]
        if not valid_results:
            print("Exiting, as no valid results were provided.")

        # Iterate through results, attempting evaluation
        for k, v in self.results_data.items():
            print("Running: %s" % valid_results)
