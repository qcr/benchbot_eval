from __future__ import print_function

import inspect
import json
import os
import pprint
import re
import numpy as np
import subprocess
import sys
import warnings
import zipfile

from .omq import OMQ
from . import class_list as cl

# Needed to simply stop it printing the source code text with the warning...
warnings.formatwarning = (lambda msg, cat, fn, ln, line: "%s:%d: %s: %s\n" %
                          (fn, ln, cat.__name__, msg))


class Evaluator:
    _TYPE_SEMANTIC_SLAM = 'semantic_slam'
    _TYPE_SCD = 'scd'

    _REQUIRED_RESULTS_STRUCTURE = {
        'task_details': {
            'type': (lambda value: value in
                     [Evaluator._TYPE_SEMANTIC_SLAM, Evaluator._TYPE_SCD]),
            'control_mode':
                lambda value: value in ['passive', 'active'],
            'localisation_mode':
                lambda value: value in ['ground_truth', 'dead_reckoning']
        },
        'environment_details': {
            'name':
                None,
            'numbers': (lambda value: type(value) is list and all(
                int(x) in range(1, 6) for x in value))
        },
        'objects': None
    }

    _REQUIRED_OBJECT_STRUCTURE = {
        'label_probs': None,
        'centroid': lambda value: len(value) == 3,
        'extent': lambda value: len(value) == 3
    }

    _REQUIRED_SCD_OBJECT_STRUCTURE = {
        'state_probs': lambda value: len(value) == 3
    }

    _ZIP_IGNORE = ["submission.json"]

    __LAMBDA_REGEX = [
        (r'Evaluator._TYPE_([^,^\]]*)', lambda x: "'%s'" % x.group(1).lower()),
        (r'  +', r' '), (r'^.*?(\(|lambda)', r'\1'), (r', *$', r''),
        (r'\n', r''), (r'^\((.*)\)$', r'\1')
    ]

    def __init__(self,
                 results_filenames,
                 ground_truth_dir,
                 scores_filename,
                 print_all=True,
                 required_task=None,
                 required_envs=None):
        # Confirm we have a valid submission file, & ground truth directory
        if not os.path.exists(ground_truth_dir):
            raise ValueError("ERROR: Ground truths directory "
                             "'%s' does not exist." % ground_truth_dir)
        for r in results_filenames:
            if not os.path.exists(r):
                raise ValueError("ERROR: Results file '%s' does not exist." %
                                 r)

        # We have valid parameters, save them & return
        self.results_filenames = results_filenames
        self.ground_truth_dir = ground_truth_dir
        self.scores_filename = scores_filename
        self.print_all = print_all
        self.required_task = required_task
        self.required_envs = required_envs

    @staticmethod
    def __lambda_to_text(l):
        s = inspect.getsource(l)
        for a, b in Evaluator.__LAMBDA_REGEX:
            s = re.sub(a, b, s, re.DOTALL)
        return s

    @staticmethod
    def __validate(data_dict, reference_dict, name):
        for k, v in reference_dict.items():
            if k not in data_dict:
                raise ValueError("Required key '%s' not found in '%s'" %
                                 (k, name))
            elif type(v) is dict and type(data_dict[k]) is not dict:
                raise ValueError("Value for '%s' in '%s' is not a dict" %
                                 (k, name))
            elif type(v) is dict:
                Evaluator.__validate(data_dict[k], v, name + "['%s']" % k)
            elif v is not None and not v(data_dict[k]):
                raise ValueError(
                    "Key '%s' in '%s' has value '%s', "
                    "which fails the check:\n\t%s" %
                    (k, name, data_dict[k], Evaluator.__lambda_to_text(v)))

    @staticmethod
    def _create_scores(task_details,
                       environment_details,
                       scores_omq,
                       scores_avg_pairwise,
                       scores_avg_label,
                       scores_avg_spatial,
                       scores_avg_fp_quality,
                       scores_avg_state_quality=None):
        return {
            'task_details': task_details,
            'environment_details': environment_details,
            'scores': {
                'OMQ':
                    scores_omq,
                'avg_pairwise':
                    scores_avg_pairwise,
                'avg_label':
                    scores_avg_label,
                'avg_spatial':
                    scores_avg_spatial,
                'avg_fp_quality':
                    scores_avg_fp_quality,
                **({} if scores_avg_state_quality is None else {
                       'avg_state_quality': scores_avg_state_quality
                   })
            }
        }

    @staticmethod
    def _evaluate_scd(results_data, ground_truth_data):
        # Takes in results data from a BenchBot submission and evaluates the
        # difference map to results

        # Use the ground truth object-based semantic maps for each scene to
        # derive the ground truth scene change semantic map (use empty lists to
        # handle missing fields just in case)
        # NOTE: ground truth uses a flag to determine change state rather than
        # the distribution provided in the submission results
        es = Evaluator._get_env_strings(results_data['environment_details'])
        gt_objects_1 = (ground_truth_data[es[0]]['objects']
                        if 'objects' in ground_truth_data[es[0]] else [])
        gt_objects_2 = (ground_truth_data[es[1]]['objects']
                        if 'objects' in ground_truth_data[es[1]] else [])
        gt_changes = [{
            **o, 'state': 'removed'
        } for o in gt_objects_1 if o not in gt_objects_2]
        gt_changes += [{
            **o, 'state': 'added'
        } for o in gt_objects_2 if o not in gt_objects_1]

        # Grab an evaluator instance, & use it to return some results
        evaluator = OMQ(scd_mode=True)
        return Evaluator._create_scores(
            task_details=results_data['task_details'],
            environment_details=results_data['environment_details'],
            scores_omq=evaluator.score([(gt_changes, results_data['objects'])
                                       ]),
            scores_avg_pairwise=evaluator.get_avg_overall_quality_score(),
            scores_avg_label=evaluator.get_avg_label_score(),
            scores_avg_spatial=evaluator.get_avg_spatial_score(),
            scores_avg_fp_quality=evaluator.get_avg_fp_score(),
            scores_avg_state_quality=evaluator.get_avg_state_score())

    @staticmethod
    def _evaluate_semantic_slam(results_data, ground_truth_data):
        # Takes in results data from a BenchBot submission, evaluates the
        # result using the ground truth data, & then spits out a dict of scores
        # data

        # Get ground truth objects from the correct ground truth set
        gt_data = ground_truth_data[Evaluator._get_env_string(
            results_data['environment_details'])]
        gt_objects = (gt_data['objects'] if 'objects' in gt_data else [])

        # Grab an evaluator instance, & use it to return some results
        evaluator = OMQ()
        return Evaluator._create_scores(
            task_details=results_data['task_details'],
            environment_details=results_data['environment_details'],
            scores_omq=evaluator.score([(gt_objects, results_data['objects'])
                                       ]),
            scores_avg_pairwise=evaluator.get_avg_overall_quality_score(),
            scores_avg_label=evaluator.get_avg_label_score(),
            scores_avg_spatial=evaluator.get_avg_spatial_score(),
            scores_avg_fp_quality=evaluator.get_avg_fp_score())

    @staticmethod
    def _get_task_string(task_details):
        return "%s:%s:%s" % (task_details['type'],
                             task_details['control_mode'],
                             task_details['localisation_mode'])

    @staticmethod
    def _get_env_string(environment_details):
        return ":".join([environment_details['name']] +
                        [str(x) for x in environment_details['numbers']])

    @staticmethod
    def _get_env_strings(environment_details):
        # Returns a list of strings referring to an individual environment
        # scene
        return ([
            "%s:%s" % (environment_details['name'], i)
            for i in environment_details['numbers']
        ])

    @staticmethod
    def _ground_truth_file(ground_truth_dir, name, number):
        filename = subprocess.check_output(
            "find %s -name '%s_%s.json'" % (ground_truth_dir, name, number),
            shell=True).decode(sys.stdout.encoding).split('\n')[0]
        if not os.path.exists(filename):
            raise ValueError(
                "Results request a ground truth for variation "
                "#%d of environment '%s', but a corresponding ground truth "
                "file (%s) could not be found in '%s'." %
                (number, name, filename, ground_truth_dir))
        return filename

    @staticmethod
    def _load_ground_truth_data(ground_truth_dir, envs_details_list):
        # Takes a list of envs, & loads the associated ground truth files
        gtd = {}  # Dict of ground truth data, with env_string as keys
        for e in envs_details_list:
            env_strs = Evaluator._get_env_strings(e)
            for i, s in zip(e['numbers'], env_strs):
                if s not in gtd:
                    fn = Evaluator._ground_truth_file(ground_truth_dir,
                                                      e['name'], i)
                    print("Loading ground truth data from '%s' ..." % fn)
                    with open(fn, 'r') as f:
                        # NOTE should remove format step in time
                        gtd[s] = Evaluator._sanitise_ground_truth(
                            (json.load(f)))
                    print("\tDone.")
        return gtd

    @staticmethod
    def _load_results_data(results_filenames):
        # Takes a list of filenames & pulls all data from JSON & *.zip files
        # (sorry... nesting abomination...)
        results = {}  # Dict of provided data, with filenames as keys
        for r in results_filenames:
            print("Loading data from '%s' ..." % r)
            if zipfile.is_zipfile(r):
                with zipfile.ZipFile(r, 'r') as z:
                    for f in z.filelist:
                        if f.filename in Evaluator._ZIP_IGNORE:
                            print("\tIgnoring file '%s'" % f.filename)
                        else:
                            with z.open(f, 'r') as zf:
                                d = None
                                try:
                                    d = json.load(zf)
                                except:
                                    print("\tSkipping file '%s'" % f.filename)
                                    continue  # Failure is fine / expected here!
                                print("\tExtracting data from file '%s'" %
                                      f.filename)
                                results[z.filename + ':' + f.filename] = (
                                    Evaluator.sanitise_results_data(d))
            else:
                with open(r, 'r') as f:
                    results[r] = Evaluator.sanitise_results_data(json.load(f))
        print("\tDone.")
        return results

    @staticmethod
    def _sanitise_ground_truth(ground_truth_data):
        # This code is only needed as we have a discrepancy between the format
        # of ground_truth_data produced in ground truth generation, & what the
        # evaluation process expects. Long term, the discrepancy should be
        # rectified & this code removed.
        for o in ground_truth_data['objects']:
            o['class_id'] = cl.get_nearest_class_id(
                o.pop('class'))  # swap name for ID
        return ground_truth_data

    @staticmethod
    def _validate_object_data(object_data, object_number, scd=False):
        # Validates whether an object has all of the required fields
        try:
            Evaluator.__validate(object_data,
                                 Evaluator._REQUIRED_OBJECT_STRUCTURE,
                                 'object')
            if scd:
                Evaluator.__validate(object_data,
                                     Evaluator._REQUIRED_SCD_OBJECT_STRUCTURE,
                                     'object')
        except Exception as e:
            raise ValueError("Validation of object #%s failed: %s" %
                             (object_number, e))

    @staticmethod
    def _validate_results_data(results_data):
        # Validates whether a results dict has all of the required fields
        try:
            Evaluator.__validate(results_data,
                                 Evaluator._REQUIRED_RESULTS_STRUCTURE,
                                 'results_data')
        except Exception as e:
            raise ValueError("Results validation failed: %s" % e)

    @staticmethod
    def _validate_results_set(results_set,
                              required_task=None,
                              required_envs=None):
        # Validates whether a set of results meets requirements as a set (i.e.
        # all tasks the same, & possible task / environment contstraints)
        task_str = required_task
        env_strs = []
        for f, d in results_set.items():
            s = Evaluator._get_task_string(d['task_details'])
            if task_str is None:
                task_str = s
            elif s != task_str and required_task is None:
                raise ValueError(
                    "JSON result files can only be evaluated together if "
                    "they are for the same task. File '%s' was for task '%s', "
                    "whereas file '%s' was for task '%s'." %
                    (results_set[0][0], task_str, f, s))
            elif s != task_str:
                raise ValueError(
                    "Evaluator was configured only accept results for task "
                    "'%s', but results file '%s' is for task '%s'" %
                    (required_task, f, s))

            env_strs.append(Evaluator._get_env_string(
                d['environment_details']))
            if (required_envs is not None and
                    env_strs[-1] not in required_envs):
                raise ValueError(
                    "Evaluator was configured to require environments: %s. "
                    "Results file '%s' is for environment '%s' which is not "
                    "in the list." %
                    (", ".join(required_envs), f, env_strs[-1]))

        # Lastly, ensure we have all required environments if relevant
        if required_envs is not None:
            for e in required_envs:
                if e not in env_strs:
                    raise ValueError(
                        "Evaluator was configured to require environments: "
                        "%s. No result was found for environment '%s'." %
                        (", ".join(required_envs), e))

    @staticmethod
    def sanitise_results_data(results_data):
        is_scd = results_data['task_details']['type'] == Evaluator._TYPE_SCD

        # Validate the provided results data
        Evaluator._validate_results_data(results_data)
        for i, o in enumerate(results_data['objects']):
            Evaluator._validate_object_data(o, i, scd=is_scd)

        # Use the default class_list if none is provided
        if 'class_list' not in results_data or not results_data['class_list']:
            warnings.warn(
                "No 'class_list' field provided; assuming results have used "
                "our default class list")
            results_data['class_list'] = cl.CLASS_LIST

        # Sanitise all probability distributions for labels & states if
        # applicable (sanitising involves dumping unused bins to the background
        # / uncertain class, normalising the total probability to 1, &
        # optionally rearranging to match a required order)
        for o in results_data['objects']:
            if len(o['label_probs']) != len(results_data['class_list']):
                raise ValueError(
                    "The label probability distribution for object %d has a "
                    "different length (%d) \nto the used class list (%d). " %
                    (i, len(o['label_probs']), len(
                        results_data['class_list'])))
            o['label_probs'] = Evaluator.sanitise_prob_dist(
                o['label_probs'], results_data['class_list'])
            if is_scd:
                o['state_probs'] = Evaluator.sanitise_prob_dist(
                    o['state_probs'])

        # We have applied our default class list to the label probs, so update
        # the class list in results_data
        results_data['class_list'] = cl.CLASS_LIST

        return results_data

    @staticmethod
    def sanitise_prob_dist(prob_dist, current_class_list=None):
        # This code makes the assumption that the last bin is the background /
        # "I'm not sure" class (it is an assumption because this function can
        # be called with no explicit use of a class list)
        BACKGROUND_CLASS_INDEX = -1

        # Create a new prob_dist if we were given a current class list by
        # converting all current classes to items in our current class
        # list, & amalgamating all duplicate values (e.g. anything not
        # found in our list will be added to the background class)
        if current_class_list is not None:
            new_prob_dist = [0.0] * len(cl.CLASS_LIST)
            for i, c in enumerate(current_class_list):
                new_prob_dist[BACKGROUND_CLASS_INDEX if cl.
                              get_nearest_class_id(c) is None else cl.
                              get_nearest_class_id(c)] += prob_dist[i]
            prob_dist = new_prob_dist

        # Either normalize the distribution if it has a total > 1, or dump
        # missing probability into the background / "I'm not sure" class
        total_prob = np.sum(prob_dist)
        if total_prob > 1:
            prob_dist /= total_prob
        else:
            prob_dist[BACKGROUND_CLASS_INDEX] += 1 - total_prob

        return prob_dist

    def evaluate(self):
        # Iteratively load data from each results file (turning *.zips into a
        # list of JSON results), & sanitise the data
        print("LOADING REQUIRED DATA FOR %d PROVIDED FILES:\n" %
              len(self.results_filenames))
        results_set = Evaluator._load_results_data(self.results_filenames)

        # Ensure the results set meets any requirements that may exist (all
        # must be same task type, may have to be a required task type, may have
        # to match a required list of environments)
        Evaluator._validate_results_set(results_set, self.required_task,
                                        self.required_envs)

        # Try & load all of the requested ground truth maps (failing loudly if
        # a required ground truth can't be found)
        ground_truth_data = Evaluator._load_ground_truth_data(
            self.ground_truth_dir,
            [r['environment_details'] for r in results_set.values()])
        print('\n' + '-' * 80 + '\n')

        # Iteratively evaluate each of the results JSONs provided, saving the
        # scores so we can amalgamate them after
        scores_data = []
        for f, d in results_set.items():
            print("EVALUATING PERFORMANCE OF RESULTS IN '%s':\n" % f)

            # Perform evaluation, selecting the appropriate evaluation function
            scores_data.append(
                (self._evaluate_scd
                 if d['task_details']['type'] == Evaluator._TYPE_SCD else
                 self._evaluate_semantic_slam)(d, ground_truth_data))

            # Print the results if allowed, otherwise just say we're done
            if self.print_all:
                print("\nScores for '%s':\n" % f)
                pprint.pprint(scores_data[-1])
            else:
                print("Done")
            print('\n' + '-' * 80 + '\n')

        # Amalgamate all of the produced scores
        scores = Evaluator._create_scores(
            task_details=scores_data[0]['task_details'],
            environment_details=[
                s['environment_details'] for s in scores_data
            ],
            scores_omq=np.mean([s['scores']['OMQ'] for s in scores_data]),
            scores_avg_pairwise=np.mean(
                [s['scores']['avg_pairwise'] for s in scores_data]),
            scores_avg_label=np.mean(
                [s['scores']['avg_label'] for s in scores_data]),
            scores_avg_spatial=np.mean(
                [s['scores']['avg_spatial'] for s in scores_data]),
            scores_avg_fp_quality=np.mean(
                [s['scores']['avg_fp_quality'] for s in scores_data]),
            scores_avg_state_quality=(np.mean([
                s['scores']['avg_state_quality'] for s in scores_data
            ]) if 'avg_state_quality' in scores_data[0]['scores'] else None),
        )

        # Print the results, save them, & finish
        print(("\nFinal scores for the '%s:%s:%s' task:\n" %
               (scores['task_details']['type'],
                scores['task_details']['control_mode'],
                scores['task_details']['localisation_mode'])).upper())
        pprint.pprint(scores)
        with open(self.scores_filename, 'w') as f:
            json.dump(scores, f)
        print("\nDone.")
