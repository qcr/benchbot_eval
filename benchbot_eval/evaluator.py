from __future__ import print_function

import json
import os
import pprint
import numpy as np
import warnings
from .cdq3d import CDQ3D
from . import class_list as cl


class Evaluator:
    _TYPE_SEMANTIC_SLAM = 'semantic_slam'
    _TYPE_SCD = 'scd'
    _IOU_THRESHOLDS = [
        0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
        0.9, 0.95
    ]

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
    def _evaluate_scd(results_data, ground_truth1_data, ground_truth2_data):
        # Takes in results data from a BenchBot submission, evaluates the
        # result using the ground truth data for env 1 and env 2, & then spits out a dict of scores data

        gt_dicts_1 = ground_truth1_data['objects']
        gt_dicts_2 = ground_truth2_data['objects']

        # Establish ground-truth scene change info from two ground-truth files
        # removed (rem) and added (add)
        # NOTE Currently assuming the gt dict will be the same across the two
        gt_dicts_rem = [
            gt_dict for gt_dict in gt_dicts_1 if gt_dict not in gt_dicts_2
        ]
        gt_dicts_add = [
            gt_dict for gt_dict in gt_dicts_2 if gt_dict not in gt_dicts_1
        ]

        # Extract the added and removed detections from the result data
        det_dicts_all = results_data['detections']
        det_dicts_rem = [
            det_dict for det_dict in det_dicts_all
            if 'changed_state' in det_dict.keys() and
            det_dict['changed_state'] == 'removed'
        ]
        det_dicts_add = [
            det_dict for det_dict in det_dicts_all
            if 'changed_state' in det_dict.keys() and
            det_dict['changed_state'] == 'added'
        ]

        evaluator = CDQ3D()

        # Get the CDQ3D scores for the removed and added maps
        scores_rem = {'CDQ3D': evaluator.score([(gt_dicts_rem, det_dicts_rem)]),
                      'avg_pairwise': evaluator.get_avg_overall_quality_score(),
                      'avg_label': evaluator.get_avg_label_score(),
                      'avg_spatial': evaluator.get_avg_spatial_score(),
                      'avg_fp_quality': evaluator.get_avg_fp_score()}
        scores_add = {'CDQ3D': evaluator.score([(gt_dicts_add, det_dicts_add)]),
                      'avg_pairwise': evaluator.get_avg_overall_quality_score(),
                      'avg_label': evaluator.get_avg_label_score(),
                      'avg_spatial': evaluator.get_avg_spatial_score(),
                      'avg_fp_quality': evaluator.get_avg_fp_score()}

        # Taking overall score average across removed and added for now
        avg_scores = {key: np.mean(scores_rem[key], scores_add[key]) for key in scores_rem}

        return {
            'task_details': results_data['task_details'],
            'environment_details': results_data['environment_details'],
            'scores': avg_scores
        }

    @staticmethod
    def _evaluate_semantic_slam(results_data, ground_truth_data):
        # Takes in results data from a BenchBot submission, evaluates the
        # result using the ground truth data, & then spits out a dict of scores data

        # NOTE currently assume there is a detections field in the results_data and
        # an objects field in the ground_truth_data
        det_dicts = results_data['detections']

        gt_dicts = ground_truth_data['objects']

        evaluator = CDQ3D()

        scores = {'CDQ3D': evaluator.score([(gt_dicts, det_dicts)]),
                  'avg_pairwise': evaluator.get_avg_overall_quality_score(),
                  'avg_label': evaluator.get_avg_label_score(),
                  'avg_spatial': evaluator.get_avg_spatial_score(),
                  'avg_fp_quality': evaluator.get_avg_fp_score()}

        return {
            'task_details': results_data['task_details'],
            'environment_details': results_data['environment_details'],
            'scores': scores
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

    def _validate_environment(self, results_data):
        if ('environment_details' not in results_data or
                'name' not in results_data['environment_details'] or
                'numbers' not in results_data['environment_details']):
            self._format_error(
                "Could not access required field "
                "results_data['environment_details']['name|numbers'].")
        # Check that all environment numbers are in the correct range
        elif (len([
                num for num in results_data['environment_details']['numbers']
                if int(num) not in range(1, 6)
        ])):
            self._format_error(
                "results_data['environment_details']['numbers'] "
                "is not in the range 1-5 (inclusive).")

    def _validate_type(self, results_data):
        if ('task_details' not in results_data or
                'type' not in results_data['task_details']):
            self._format_error("Could not access required field "
                               "results_data['task_details']['type'].")
        elif (results_data['task_details']['type'] !=
              Evaluator._TYPE_SEMANTIC_SLAM and
              results_data['task_details']['type'] != Evaluator._TYPE_SCD):
            self._format_error(
                "results_data['task_details']['type'] is "
                "not '%s' or '%s'. No other modes are supported." %
                (Evaluator._TYPE_SEMANTIC_SLAM, Evaluator._TYPE_SCD))

    @staticmethod
    def _format_gt(gt_data):
        # This should only be temporary code assuming we keep the current evaluation process and GT format
        for gt_dict in gt_data['objects']:
            # change classes by name to class ids based on CLASS_LIST
            gt_dict['class_id'] = cl.CLASS_IDS[gt_dict.pop('class')]
        return gt_data

    @staticmethod
    def _format_results_data(result_data):
        if 'detections' not in result_data:
            raise KeyError('Results dictionary does not have a "detections" key')
        if 'class_list' not in result_data:
            warnings.warn('class_list not provided in result_data, assuming default class list')
            result_class_list = cl.CLASS_LIST
        else:
            result_class_list = result_data['class_list']

        # check result detections have a prob_dist key and content is formatted correctly
        for det_id, det_dict in enumerate(result_data['detections']):
            if 'prob_dist' not in det_dict.keys():
                raise KeyError('Detection {} does not have a "prob_dist" key'.format(det_id))

            if len(det_dict['prob_dist']) != len(result_class_list):
                raise ValueError('Probability distributioin for detection {} has incorrect size.\n'
                                 'Is {} but should match your defined class list size ({})\n'
                                 'Note, the final class is background.'
                                 ''.format(det_id, len(det_dict['prob_dist']), len(result_class_list)))

            # Format the probability distribution to match the class list order used for evaluation
            # Work out which of the submission classes correspond to which of our classes

            det_dict['prob_dist'] = Evaluator._format_prob_dist(result_class_list, det_dict['prob_dist'])

    @staticmethod
    def _format_prob_dist(original_class_list, original_prob_dist):
        # Find the corresponding indices for every entry in the probability distribution
        eval_class_ids = []
        original_class_ids = []
        for sub_class_id, class_name in enumerate(original_class_list):
            our_class_id = cl.get_class_id(class_name)
            if our_class_id is not None:
                eval_class_ids.append(our_class_id),
                original_class_ids.append(sub_class_id)

        # Use numpy list indexing to move specific indexes from the submission
        eval_prob_list = np.zeros(len(cl.CLASS_LIST), dtype=np.float32)
        eval_prob_list[eval_class_ids] = np.array(original_prob_dist)[original_class_ids]

        # Normalize any distribution with a total greater than 1 (no all 1 distributions)
        total_prob = np.sum(eval_prob_list)
        if total_prob > 1:
            eval_prob_list /= total_prob

        # return the updated probability distribution as a list
        return eval_prob_list.tolist()


    def evaluate(self):
        # Open the submission & attempt to perform evaluation
        print("Evaluating the performance from results in: %s\n" %
              self.results_filename)
        with open(self.results_filename, 'r') as f:
            # Pull in the json data, ensuring a valid type is provided
            results_data = json.load(f)
            self._validate_type(results_data)

        # Get the ground_truth_data by using the environment fields in
        # results_data
        self._validate_environment(results_data)
        Evaluator._format_results_data(results_data)
        ground_truth_data_all = []
        for number in results_data['environment_details']['numbers']:
            with open(
                    self._ground_truth_file(
                        results_data['environment_details']['name'], number),
                    'r') as f:
                # NOTE should remove format step in time
                ground_truth_data_all.append(Evaluator._format_gt((json.load(f))))

        # Perform evaluation
        if results_data['task_details']['type'] == Evaluator._TYPE_SCD:
            # Check we have two sets of ground-truth data
            if len(ground_truth_data_all) != 2:
                raise ValueError("Scene Change Detection requires exactly"
                                 "2 environments. {} provided".format(
                                     len(ground_truth_data_all)))
            scores_data = self._evaluate_scd(results_data,
                                             ground_truth_data_all[0],
                                             ground_truth_data_all[1])
        else:
            # Check we have only one set of ground-truth data
            if len(ground_truth_data_all) != 1:
                raise ValueError("Semantic SLAM requires exactly"
                                 "1 environment. {} provided".format(
                                     len(ground_truth_data_all)))
            scores_data = self._evaluate_semantic_slam(
                results_data, ground_truth_data_all[0])

        # Print the results cutely?
        pprint.pprint(scores_data)

        # Save the results
        with open(self.scores_filename, 'w') as f:
            json.dump(scores_data, f)

        print("\nDone.")
