import pickle

from . import helpers


class Evaluator:
    def __init__(self,
                 evaluation_method_filename,
                 ground_truths_filenames,
                 skip_load=False):
        self.evaluation_method_data = helpers.load_yaml(
            evaluation_method_filename)
        self.ground_truths_data = helpers.load_yaml_list(
            ground_truths_filenames)

        self.formats_data = None
        self.ground_truth_data = None
        self.results_data = None

        self.required_task = None
        self.required_envs = None

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
