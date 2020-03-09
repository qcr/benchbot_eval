from __future__ import print_function

import json
import os
import pprint
import sys


class Evaluator:
    _TYPE_SEMANTIC_SLAM = 'semantic_slam'
    _TYPE_SCD = 'scd'

    def __init__(self, results_filename, ground_truth_dir, scores_filename):
        # Confirm we have a valid submission file, & ground truth directory
        if not os.path.exists(ground_truth_dir):
            raise ValueError("ERROR: Ground truths directory "
                             "'%s' does not exist." % ground_truth_dir)
        if not os.path.exists(results_filename):
            raise ValueError("ERROR: Results file '%s' does not exist." %
                             results_filename)

        # We have valid parameters, save them & return
        self.results_filename = results_filename
        self.ground_truth_dir = ground_truth_dir
        self.scores_filename = scores_filename

    @staticmethod
    def _evaluate_scd(results_json, ground_truth_json):
        # Takes in a results json from a BenchBot submission, evaluates the
        # result using the ground truth json, & then spits out a json
        return {
            'task_details': results_json['task_details'],
            'environment_details': results_json['environment_details'],
            'scores': {
                'map3D': 1000,
                'mapbev': 1000
            }
        }

    @staticmethod
    def _evaluate_semantic_slam(results_json, ground_truth_json):
        # Takes in a results json from a BenchBot submission, evaluates the
        # result using the ground truth json, & then spits out a json
        return {
            'task_details': results_json['task_details'],
            'environment_details': results_json['environment_details'],
            'scores': {
                'map3D': 10,
                'mapbev': 10
            }
        }

    def _format_error(self, description):
        raise ValueError(
            "Cannot perform evaluation on results contained in '%s'. %s" %
            (self.results_filename, description))

    def _ground_truth_file(self, name, number):
        filename = os.path.join(self.ground_truth_dir,
                                "%s_%s.json" % (name, number))
        if not os.path.exists(filename):
            raise ValueError(
                "Results request a ground truth for variation "
                "#%d of environment '%s', but a corresponding ground truth "
                "file (%s) could not be found." % (number, name, filename))
        return filename

    def _validate_environment(self, results_json):
        if ('environment_details' not in results_json or
                'name' not in results_json['environment_details'] or
                'number' not in results_json['environment_details']):
            self._format_error(
                "Could not access required field "
                "results_json['environment_details']['name|number'].")
        elif results_json['environment_details']['number'] not in range(1, 6):
            self._format_error("results_json['environment_details']['number'] "
                               "is not in the range 1-5 (inclusive).")

    def _validate_type(self, results_json):
        if ('task_details' not in results_json or
                'type' not in results_json['task_details']):
            self._format_error("Could not access required field "
                               "results_json['task_details']['type'].")
        elif (results_json['task_details']['type'] !=
              Evaluator._TYPE_SEMANTIC_SLAM and
              results_json['task_details']['type'] !=
              Evaluator._TYPE_SEMANTIC_SLAM):
            self._format_error(
                "results_json['task_details']['type'] is "
                "not '%s' or '%s'. No other modes are supported." %
                (Evaluator._TYPE_SEMANTIC_SLAM, Evaluator._TYPE_SCD))

    def evaluate(self):
        # Open the submission & attempt to perform evaluation
        print("Evaluating the performance from results in: %s\n" %
              self.results_filename)
        with open(self.results_filename, 'r') as f:
            # Pull in the json, ensuring a valid type is provided
            results_json = json.load(f)
            self._validate_type(results_json)

        # Get the ground_truth_json by using the environment fields in
        # results_json
        self._validate_environment(results_json)
        with (open(
                self._ground_truth_file(**results_json['environment_details']),
                'r')) as f:
            ground_truth_json = json.load(f)

        # Perform evaluation
        scores_fn = (self._evaluate_scd if results_json['task_details']['type']
                     == Evaluator._TYPE_SCD else self._evaluate_semantic_slam)
        scores_json = scores_fn(results_json, ground_truth_json)

        # Print the results cutely?
        pprint.pprint(scores_json)

        # Save the results
        with open(self.scores_filename, 'w') as f:
            json.dump(scores_json, f)

        print("\nDone.")
