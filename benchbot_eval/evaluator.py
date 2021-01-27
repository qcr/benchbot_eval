import pickle

from benchbot_addons import manager as bam

from . import helpers


class Evaluator:
    def __init__(self, evaluation_method, skip_load=False):
        self.evaluation_method_data = bam.get_match(
            "evaluation_methods", [("name", evaluation_method)],
            return_data=True)

        self.evaluate_fns = bam.load_functions(self.evaluation_method_data)
        assert 'evaluate' in self.evaluate_fns.keys(), (
            "No 'evaluate' function was found when loading method '%s'",
            evaluation_method)
        self.evaluate_fn = self.evaluate_fns['evaluate']

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

        # Load available ground truth data
        # TODO ideally this would only load what is required, but the manager
        # in benchbot_addons would need to support nested 'get_match' queries
        gt_data = bam.load_yaml_list(bam.find_all("ground_truths", 'json'))

        # Iterate through results, attempting evaluation
        gt_formats = self.evaluation_method_data['valid_ground_truth_formats']
        scores_data = []
        for k, v in self.results_data.items():
            # Find a usable ground truth (one of supported formats, & for same
            # environment as this result
            gts = []
            for e in v['environment_details']:
                gt = next(
                    (g for g in gt_data
                     if bam.env_string(g['environment']) == bam.env_string(e)
                     and g['format']['name'] in gt_formats), None)
                assert gt is not None, (
                    "No ground truth could was found for '%s' with "
                    "any of the following formats:\n\t%s" %
                    (bam.env_string(e), gt_formats))
                gts.append(gt)

            # Score the result
            scores_data.append(self.evaluate_fn(v, gts))

            # Print the results (if allowed)

        # Amalgamate all of the produced scores if supported by evaluation
        # method
