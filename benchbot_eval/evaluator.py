from __future__ import print_function

import json
import os
import pprint
import sys
import numpy as np
from .iou_tools import IoU


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
    def _compute_map(gt_dicts, det_dicts):
        """
        Calculate all map score summaries for given set of ground-truth objects and detections.
        Here ground-truth objects and detections are represented with dictionaries
        """
        # Compile collection of all classes of ground-truth objects
        # NOTE we do not care about if detection has an unkown class (all such detections would be FPs)
        object_classes = list(np.unique([gt_dict['class'] for gt_dict in gt_dicts]))
        iou_calculator = IoU()

        total_mAP_2d = 0.0
        total_mAP_3d = 0.0
        mAP_3d_25 = 0.0
        mAP_2d_25 = 0.0
        mAP_3d_50 = 0.0
        mAP_2d_50 = 0.0
        
        # Evaluate AP for each class
        for object_class in object_classes:
            # Get the detections for the current class (sorted by highest confidence)
            class_dets = [det_dict for det_dict in det_dicts if det_dict['class'] == object_class]
            class_dets = sorted(class_dets, key=lambda conf: conf['confidence'], reverse=True)

            # Get the ground-truth instances for the current class
            class_gts = [gt_dict for gt_dict in gt_dicts if gt_dict['class'] == object_class]

            # Matrix holding the IoU values for 2D and 3D situation for each GT and Det
            ious_2d = np.ones((len(class_gts), len(class_dets)))   # G x D
            ious_3d = np.ones((len(class_gts), len(class_dets)))   # G x D
            
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
            for threshold in Evaluator._IOU_THRESHOLDS:
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

                mAP_2d = Evaluator._compute_ap(tps_2d, fps_2d, len(class_gts))
                mAP_3d = Evaluator._compute_ap(tps_3d, fps_3d, len(class_gts))

                total_mAP_2d += mAP_2d
                total_mAP_3d += mAP_3d

                if threshold == 0.25:
                    mAP_3d_25 = mAP_3d
                    mAP_2d_25 = mAP_2d
                elif threshold == 0.5:
                    mAP_3d_50 = mAP_3d
                    mAP_2d_50 = mAP_2d
        
        # Average over the number of classes to get final mAP score
        mAP_2d = total_mAP_2d / (len(object_classes)*len(Evaluator._IOU_THRESHOLDS))
        mAP_3d = total_mAP_3d / (len(object_classes)*len(Evaluator._IOU_THRESHOLDS))

        return {
                'mAP3D': mAP_3d,
                'mAPbev': mAP_2d,
                'mAP3D_25': mAP_3d_25,
                'mAP3D_50': mAP_3d_50,
                'mAPbev_25': mAP_2d_25,
                'mAPbev_50': mAP_2d_50,
            }

    @staticmethod
    def _evaluate_scd(results_data, ground_truth1_data, ground_truth2_data):
        # Takes in results data from a BenchBot submission, evaluates the
        # result using the ground truth data for env 1 and env 2, & then spits out a dict of scores data
        
        gt_dicts_1 = ground_truth1_data['objects']
        gt_dicts_2 = ground_truth2_data['objects']
        
        # Establish ground-truth scene change info from two ground-truth files
        # removed (rem) and added (add)
        # NOTE Currently assuming the gt dict will be the same across the two
        gt_dicts_rem = [gt_dict for gt_dict in gt_dicts_1 if gt_dict not in gt_dicts_2]
        gt_dicts_add = [gt_dict for gt_dict in gt_dicts_2 if gt_dict not in gt_dicts_1]

        # Extract the added and removed detections from the result data
        det_dicts_all = results_data['detections']
        det_dicts_rem = [det_dict for det_dict in det_dicts_all 
                         if 'changed_state' in det_dict.keys() and 
                         det_dict['changed_state'] == 'removed']
        det_dicts_add = [det_dict for det_dict in det_dicts_all 
                         if 'changed_state' in det_dict.keys() and 
                         det_dict['changed_state'] == 'added']
        
        # Get the mAP scores for the removed and added maps
        scores_rem = Evaluator._compute_map(gt_dicts_rem, det_dicts_rem)
        scores_add = Evaluator._compute_map(gt_dicts_add, det_dicts_add)

        # Calculate weighted average of removed and added scores
        mAP3d_rem = scores_rem['mAP3D']
        mAP3d_add = scores_add['mAP3D']
        mAP2d_rem = scores_rem['mAPbev']
        mAP2d_add = scores_add['mAPbev']
        ngt_add = len(gt_dicts_add)
        ngt_rem = len(gt_dicts_rem)

        # NOTE here the mAP3D an mAPbev are weighted averages of the rem and add scores
        scores = {'mAP3D_a': mAP3d_add,
                  'mAP3D_r': mAP3d_rem,
                  'mAPbev_a': mAP2d_add,
                  'mAPbev_r': mAP2d_rem,
                  'mAP3D': (ngt_add * mAP3d_add + ngt_rem * mAP3d_rem)/(ngt_add + ngt_rem),
                  'mAPbev': (ngt_add * mAP2d_add + ngt_rem * mAP2d_rem)/(ngt_add + ngt_rem)
                  }

        return {
            'task_details': results_data['task_details'],
            'environment_details': results_data['environment_details'],
            'scores': scores
        }

    @staticmethod
    def _evaluate_semantic_slam(results_data, ground_truth_data):
        # Takes in results data from a BenchBot submission, evaluates the
        # result using the ground truth data, & then spits out a dict of scores data
    
        # NOTE currently assume there is a detections field in the results_data and 
        # an objects field in the ground_truth_data
        det_dicts = results_data['detections']
        
        gt_dicts = ground_truth_data['objects']
        
        
        return {
            'task_details': results_data['task_details'],
            'environment_details': results_data['environment_details'],
            'scores': Evaluator._compute_map(gt_dicts, det_dicts)
        
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
        elif len([num for num in results_data['environment_details']['numbers'] if num not in range(1, 6)]):
            self._format_error("results_data['environment_details']['number'] "
                               "is not in the range 1-5 (inclusive).")

    def _validate_type(self, results_data):
        if ('task_details' not in results_data or
                'type' not in results_data['task_details']):
            self._format_error("Could not access required field "
                               "results_data['task_details']['type'].")
        elif (results_data['task_details']['type'] !=
              Evaluator._TYPE_SEMANTIC_SLAM and
              results_data['task_details']['type'] !=
              Evaluator._TYPE_SCD):
            self._format_error(
                "results_data['task_details']['type'] is "
                "not '%s' or '%s'. No other modes are supported." %
                (Evaluator._TYPE_SEMANTIC_SLAM, Evaluator._TYPE_SCD))

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
        ground_truth_data_all = []
        for number in results_data['environment_details']['numbers']:
            with open(self._ground_truth_file(results_data['environment_details']['name'], number),
                      'r') as f:
                ground_truth_data_all.append(json.load(f))

        # Perform evaluation
        if results_data['task_details']['type'] == Evaluator._TYPE_SCD:
            # Check we have two sets of ground-truth data
            if len(ground_truth_data_all) != 2:
                raise ValueError("Scene Change Detection requires exactly" 
                                 "2 environments. {} provided".format(len(ground_truth_data_all)))
            scores_data = self._evaluate_scd(results_data, ground_truth_data_all[0], ground_truth_data_all[1])
        else:
            # Check we have only one set of ground-truth data
            if len(ground_truth_data_all) != 1:
                raise ValueError("Semantic SLAM requires exactly" 
                                 "1 environment. {} provided".format(len(ground_truth_data_all)))
            scores_data = self._evaluate_semantic_slam(results_data, ground_truth_data_all[0])

        # Print the results cutely?
        pprint.pprint(scores_data)

        # Save the results
        with open(self.scores_filename, 'w') as f:
            json.dump(scores_data, f)

        print("\nDone.")
