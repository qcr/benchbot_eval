from __future__ import print_function

import json
import os
import pprint
import sys
import numpy as np
from iou_tools import IoU


class Evaluator:
    _TYPE_SEMANTIC_SLAM = 'semantic_slam'
    _TYPE_SCD = 'scd'
    _IOU_THRESHOLDS = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 
                       0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

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
    def _compute_ap(tps, fps, num_gt):
        """
        Adapted from https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
        Calculate the AP given the recall and precision array
        1st) We compute a version of the measured precision/recall curve with
            precision monotonically decreasing
        2nd) We compute the AP as the area under this curve by numerical integration.
        Input is tp and fp boolean vectors ordered by highest class confidence
        """
        # Check if we have any TPs at all. If not mAP is zero
        if np.sum(tps) == 0:
          return 0
        # Calculate cumulative sums
        cum_fp = np.cumsum(fps)
        cum_tp = np.cumsum(tps)  

        # Calculate the recall and precision as each detection is "introduced"
        # Note this is ordered by class confidence starting with highest confidence
        rec = cum_tp / num_gt
        prec = cum_tp / (cum_tp + cum_fp + 1e-10)

        # bound the precisions and recall values
        mrec = np.concatenate(([0.], rec, [1.]))  
        mpre = np.concatenate(([0.], prec, [0.]))  
        
        # compute the precision envelope  
        # i.e. remove the "wiggles" from the PR curve
        for i in range(mpre.size - 1, 0, -1):  
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])  

        # to calculate area under PR curve, look for points  
        # where X axis (recall) changes value  
        i = np.where(mrec[1:] != mrec[:-1])[0]  

        # and sum (\Delta recall) * prec  
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  
        return ap

    @staticmethod
    def _evaluate_semantic_slam(results_json, ground_truth_json):
        # Takes in a results json from a BenchBot submission, evaluates the
        # result using the ground truth json, & then spits out a json
    
        # NOTE currently assume there is a detections field in the results_json and ground_truth_json is just lsit of dict objects for now
        with open(results_json, 'r') as result_file:
            det_dicts = json.load(result_file)['detections']
        
        with open(ground_truth_json, 'r') as gt_file:
            gt_dicts = json.load(gt_file)
        
        # Compile collection of all classes of ground-truth objects
        # NOTE we do not care about if detection has an unkown class (all such detections would be FPs)
        object_classes = list(np.unique([gt_dict['class'] for gt_dict in gt_dicts]))
        iou_calculator = IoU()

        # TODO can possibly remove the weighted version for release
        total_mAP_2d = 0.0
        total_mAP_2d_weighted = 0.0
        total_mAP_3d = 0.0
        total_mAP_3d_weighted = 0.0
        
        # Evaluate AP for each class
        for object_class in object_classes:
            # Get the detections for the current class (sorted by highest confidence)
            class_dets = [det_dict for det_dict in det_dicts if det_dict['class'] == object_class]
            class_dets = sorted(class_dets, key=lambda conf: conf['confidence'], reverse=True)

            # Get the ground-truth instances for the current class
            class_gts = [gt_dict for gt_dict in gt_dicts if gt_dict['class'] == object_class]

            # Matrix holding the IoU values for 2D and 3D situation for each GT and Det
            ious_2d = np.ones((len(class_gts), len(class_dets)))*-1   # G x D
            ious_3d = np.ones((len(class_gts), len(class_dets)))*-1   # G x D
            
            # Calculate IoUs
            for det_id, det_dict in enumerate(class_dets):
                for gt_id, gt_dict in enumerate(class_gts):
                    ious_2d[gt_id, det_id], ious_3d[gt_id, det_id] = iou_calculator.dict_iou(gt_dict, det_dict)
            
            # Rank each ground-truth by how well it overlaps the given detection
            # Note in code we use 1 - iou to rank based on similarity
            det_assignment_rankings_2d = np.argsort((1 - ious_2d), axis=0)
            det_assignment_rankings_3d = np.argsort((1 - ious_3d), axis=0)
            
            # NOTE Should we be looking at different thresholds for 2D vs 3D?
            # Calculate mAP for each threshold level
            for threshold in _IOU_THRESHOLDS:
            assigned_2d = np.zeros(ious_2d.shape, bool)
            assigned_3d = np.zeros(ious_3d.shape, bool)
            # Perform assignments
            for det_id in range(len(class_dets)):
                
                # TODO Must be neater way to do this later!
                # 2D assignment
                for gt_id in det_assignment_rankings_2d[:,det_id]:
                # If we are lower than the iou threshold we aren't going to find a match
                if ious_2d[gt_id, det_id] < threshold:
                    break
                # If we are higher than the threshold and gt not assigned, assign it and move on
                elif not np.any(assigned_2d[gt_id, :]):
                    assigned_2d[gt_id, det_id] = True
                    break
                
                # 3D assignment
                for gt_id in det_assignment_rankings_3d[:,det_id]:
                # If we are lower than the iou threshold we aren't going to find a match
                if ious_3d[gt_id, det_id] < threshold:
                    break
                # If we are higher than the threshold and gt not assigned, assign it and move on
                elif not np.any(assigned_3d[gt_id, :]):
                    assigned_3d[gt_id, det_id] = True
                    break
            
            # Condense to TPs and FPs for detections
            # NOTE using sum as there should be only one for each column (quicker than non_zero or bool?)
            tps_2d = np.sum(assigned_2d, axis=0).astype(bool)
            tps_3d = np.sum(assigned_3d, axis=0).astype(bool)
            fps_2d = np.logical_not(tps_2d)
            fps_3d = np.logical_not(tps_3d)

            mAP_2d = self._compute_ap(tps_2d, fps_2d, len(class_gts))
            mAP_3d = self._compute_ap(tps_3d, fps_3d, len(class_gts))

            total_mAP_2d += mAP_2d
            total_mAP_3d += mAP_3d

            # TODO Can possibly remove this? Double check it is correct
            # multiplying by number of ground-truth objects for the class
            total_mAP_2d_weighted += mAP_2d * len(class_gts)
            total_mAP_3d_weighted += mAP_3d * len(class_gts)
        
        # Average over the number of classes to get final mAP score
        mAP_2d = total_mAP_2d / (len(object_classes)*len(_IOU_THRESHOLDS))
        mAP_3d = total_mAP_3d / (len(object_classes)*len(_IOU_THRESHOLDS))
        mAP_2d_weighted = total_mAP_2d_weighted / (len(object_classes)*len(_IOU_THRESHOLDS))
        mAP_3d_weighted = total_mAP_3d_weighted / (len(object_classes)*len(_IOU_THRESHOLDS))

        return {
            'task_details': results_json['task_details'],
            'environment_details': results_json['environment_details'],
            'scores': {
                'map3D': mAP_3d,
                'mapbev': mAP_2d
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
