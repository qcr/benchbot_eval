import pickle

from benchbot_addons import manager as bam

from . import helpers


class Evaluator:
    def __init__(self, evaluation_method, skip_load=False):
        self.evaluation_method_data = bam.get_match(
            "evaluation_methods", [("name", evaluation_method)],
            return_data=True)

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
        supported_formats = self.evaluation_method_data[
            'valid_ground_truth_formats']
        for k, v in self.results_data.items():
            # Find a usable ground truth (one of supported formats, & for same
            # environment as this result
            gt = next((g for g in self.ground_truth_data
                       if helpers.env_string(g['environment_details']) ==
                       helpers.env_string(v['environment_details']) and g['']),
                      None)

            # Score the result

            # Print the results (if allowed)

        # Amalgamate all of the produced scores if supported by evaluation
        # method
