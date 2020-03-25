from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import gmean
from . import iou_tools

_IOU_TOOL = iou_tools.IoU()

# NOTE For now we will ignore the concept of foreground and background quality in favor of
# spatial quality being just the IoU of a detection.


class CDQ3D(object):
    """
    Class for evaluating results from a Semantic SLAM system.
    Based upon the PDQ system found below
    https://github.com/david2611/pdq_evaluation
    """
    def __init__(self):
        """
        Initialisation function for Semantic SLAM evaluator, classical detection quality 3D (CDQ3D).
        """
        super(CDQ3D, self).__init__()
        self._tot_overall_quality = 0.0
        self._tot_spatial_quality = 0.0
        self._tot_label_quality = 0.0
        self._tot_fp_cost = 0.0
        self._tot_TP = 0
        self._tot_FP = 0
        self._tot_FN = 0
        self._det_evals = []
        self._gt_evals = []

    def reset(self):
        """
        Reset all internally stored evaluation measures to zero.
        :return: None
        """
        self._tot_overall_quality = 0.0
        self._tot_spatial_quality = 0.0
        self._tot_label_quality = 0.0
        self._tot_fp_cost = 0.0
        self._tot_TP = 0
        self._tot_FP = 0
        self._tot_FN = 0
        self._det_evals = []
        self._gt_evals = []

    def add_map_eval(self, gt_instances, det_instances):
        """
        Adds a single map's detections and ground-truth to the overall evaluation analysis.
        :param gt_instances: list of ground-truth dictionaries present in the given map.
        :param det_instances: list of detection dictionaries objects provided for the given map
        :return: None
        """
        results = _calc_qual_map(gt_instances, det_instances)
        self._tot_overall_quality += results['overall']
        self._tot_spatial_quality += results['spatial']
        self._tot_label_quality += results['label']
        self._tot_fp_cost += results['fp_cost']
        self._tot_TP += results['TP']
        self._tot_FP += results['FP']
        self._tot_FN += results['FN']
        self._det_evals.append(results['map_det_evals'])
        self._gt_evals.append(results['map_gt_evals'])

    def get_current_score(self):
        """
        Get the current score for all scenes analysed at the current time.
        :return: The CDQ3D score across all maps as a float
        (average pairwise quality over the number of object-detection pairs observed with FPs weighted by confidence).
        """
        denominator = self._tot_TP + self._tot_fp_cost + self._tot_FN
        return self._tot_overall_quality/denominator

    def score(self, param_lists):
        """
        Calculates the average quality score for a set of detections on
        a set of ground truth objects over a series of maps.
        The average is calculated as the average pairwise quality over the number of object-detection pairs observed
        with FPs weighted by confidence.
        Note that this removes any evaluation information that had been stored for previous maps.
        Assumes you want to score just the full list you are given.
        :param param_lists: A list of tuples where each tuple holds a list of ground-truth dicts and a list of
        detection dicts. Each map observed is an entry in the main list.
        :return: The CDQ3D score across all maps as a float
        """
        self.reset()

        num_maps = len(param_lists)
        for map_params in param_lists:
            map_results = _get_map_evals(map_params)
            self._tot_overall_quality += map_results['overall']
            self._tot_spatial_quality += map_results['spatial']
            self._tot_label_quality += map_results['label']
            self._tot_fp_cost += map_results['fp_cost']
            self._tot_TP += map_results['TP']
            self._tot_FP += map_results['FP']
            self._tot_FN += map_results['FN']
            self._det_evals.append(map_results['map_det_evals'])
            self._gt_evals.append(map_results['map_gt_evals'])

        return self.get_current_score()

    def get_avg_spatial_score(self):
        """
        Get the average spatial quality score for all assigned detections in all maps analysed at the current time.
        Note that this is averaged over the number of assigned detections (TPs) and not the full set of TPs, FPs,
        and FNs like the final Semantic SLAM CDQ3D score.
        :return: average spatial quality of every assigned detection
        """
        if self._tot_TP > 0.0:
            return self._tot_spatial_quality / float(self._tot_TP)
        return 0.0

    def get_avg_label_score(self):
        """
        Get the average label quality score for all assigned detections in all maps analysed at the current time.
        Note that this is averaged over the number of assigned detections (TPs) and not the full set of TPs, FPs,
        and FNs like the final Semantic SLAM CDQ3D score.
        :return: average label quality of every assigned detection
        """
        if self._tot_TP > 0.0:
            return self._tot_label_quality / float(self._tot_TP)
        return 0.0

    def get_avg_overall_quality_score(self):
        """
        Get the average overall pairwise quality score for all assigned detections
        in all frames analysed at the current time.
        Note that this is averaged over the number of assigned detections (TPs) and not the full set of TPs, FPs,
        and FNs like the final Semantic SLAM CDQ3D score.
        :return: average overall pairwise quality of every assigned detection
        """
        if self._tot_TP > 0.0:
            return self._tot_overall_quality / float(self._tot_TP)
        return 0.0

    def get_avg_fp_score(self):
        """
        Get the average quality (1 - cost) for all false positive detections across all maps analysed at the
        current time.
        Note that this is averaged only over the number of FPs and not the full set of TPs, FPs, and FNs.
        Note that at present false positive cost/quality is based solely upon label scores.
        :return: average false positive quality score
        """
        if self._tot_FP > 0.0:
            return (self._tot_FP - self._tot_fp_cost) / self._tot_FP
        return 1.0

    def get_assignment_counts(self):
        """
        Get the total number of TPs, FPs, and FNs across all frames analysed at the current time.
        :return: tuple containing (TP, FP, FN)
        """
        return self._tot_TP, self._tot_FP, self._tot_FN


def _get_map_evals(parameters):
    """
    Evaluate the results for a given image
    :param parameters: tuple containing list of ground-truth dicts and detection dicts
    :return: results dictionary containing total overall spatial quality, total spatial quality on positively assigned
    detections, total label quality on positively assigned detections, total foreground spatial quality on positively
    assigned detections, total background spatial quality on positively assigned detections, number of true positives,
    number of false positives, number false negatives, detection evaluation summary, and ground-truth evaluation summary
    for the given image.
    Format {'overall':<tot_overall_quality>, 'spatial': <tot_tp_spatial_quality>, 'label': <tot_tp_label_quality>,
    'TP': <num_true_positives>, 'FP': <num_false_positives>, 'FN': <num_false_positives>,
    'img_det_evals':<detection_evaluation_summary>, 'img_gt_evals':<ground-truth_evaluation_summary>}
    """
    gt_instances, det_instances = parameters
    results = _calc_qual_map(gt_instances, det_instances)
    return results


def _vectorize_map_gts(gt_instances):
    """
    Vectorizes the required elements for all ground-truth dicts as necessary for a given map.
    These elements are the ground-truth boxes and the class ids
    :param gt_instances: list of all ground-truth dicts for a given image
    :return: (gt_boxes, gt_labels).
    gt_boxes: g length list of box centroids and extents stored as dictionaries for each of the g ground-truth dicts
    (dictionary format: {'centroid': <centroid>, 'extent': <extent>})
    gt_labels: g, numpy array of class labels as an integer for each of the g ground-truth dicts
    """
    gt_labels = np.array([gt_instance['class_id'] for gt_instance in gt_instances], dtype=np.int)        # g,
    gt_boxes = [{"centroid": gt_instance['centroid'], "extent": gt_instance["extent"]}
                for gt_instance in gt_instances]                                                         # g,

    return gt_boxes, gt_labels


def _vectorize_map_dets(det_instances):
    """
    Vectorize the required elements for all detection dicts as necessary for a given map.
    These elements are the detection boxes and the probability distributions.
    :param det_instances: list of all detection dicts for a given image.
    :return: (det_boxes, det_prob_mat)
    det_boxes: d length list of box centroids and extents stored as dictionaries for each of the d detections
    (dictionary format: {'centroid': <centroid>, 'extent': <extent>})
    det_label_prob_mat: d x c numpy array of label probability scores across all c classes for each of the d detections
    """
    det_prob_mat = np.stack([np.array(det_instance['prob_dist']) for det_instance in det_instances], axis=0)  # d x c
    det_boxes = [{"centroid": det_instance['centroid'], "extent": det_instance["extent"]}
                 for det_instance in det_instances]                                                           # d,
    return det_boxes, det_prob_mat


def _calc_spatial_qual(gt_boxes, det_boxes):
    """
    Calculate the spatial quality for all detections on all ground truth objects for a given map.
    :param: gt_boxes: g length list of all ground-truth box dictionaries defining centroid and extent of objects
    :param: det_boxes: d length list of all detection box dictionaries defining centroid and extent of objects
    :return: spatial_quality: g x d spatial quality score between zero and one for each possible combination of
    g ground truth objects and d detections.
    """
    # TODO optimize in some clever way in future rather than two for loops
    spatial_quality = np.zeros((len(gt_boxes), len(det_boxes)), dtype=np.float)     # g x d
    for gt_id, gt_box_dict in enumerate(gt_boxes):
        for det_id, det_box_dict in enumerate(det_boxes):
            spatial_quality[gt_id, det_id] = _IOU_TOOL.dict_iou(gt_box_dict, det_box_dict)[1]

    return spatial_quality


def _calc_label_qual(gt_label_vec, det_prob_mat):
    """
    Calculate the label quality for all detections on all ground truth objects for a given image.
    :param gt_label_vec:  g, numpy array containing the class label as an integer for each object.
    :param det_prob_mat: d x c numpy array of label probability scores across all c classes
    for each of the d detections.
    :return: label_qual_mat: g x d label quality score between zero and one for each possible combination of
    g ground truth objects and d detections.
    """
    label_qual_mat = det_prob_mat[:, gt_label_vec].T.astype(np.float32)     # g x d
    return label_qual_mat


def _calc_overall_qual(label_qual, spatial_qual):
    """
    Calculate the overall quality for all detections on all ground truth objects for a given image
    :param label_qual: g x d label quality score between zero and one for each possible combination of
    g ground truth objects and d detections.
    :param spatial_qual: g x d spatial quality score between zero and one for each possible combination of
    g ground truth objects and d detections.
    :return: overall_qual_mat: g x d overall label quality between zero and one for each possible combination of
    g ground truth objects and d detections.
    """
    combined_mat = np.dstack((label_qual, spatial_qual))

    # Calculate the geometric mean between label quality and spatial quality.
    # Note we ignore divide by zero warnings here for log(0) calculations internally.
    with np.errstate(divide='ignore'):
        overall_qual_mat = gmean(combined_mat, axis=2)

    return overall_qual_mat


def _gen_cost_tables(gt_instances, det_instances):
    """
    Generate the cost tables containing the cost values (1 - quality) for each combination of ground truth objects and
    detections within a given map.
    :param gt_instances: list of all ground-truth dicts for a given map.
    :param det_instances: list of all detection dicts for a given map.
    :return: dictionary of g x d cost tables for each combination of ground truth objects and detections.
    Note that all costs are simply 1 - quality scores (required for Hungarian algorithm implementation)
    Format: {'overall': overall summary cost table, 'spatial': spatial quality cost table,
    'label': label quality cost table}
    """
    # Initialise cost tables
    n_pairs = max(len(gt_instances), len(det_instances))
    overall_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
    spatial_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
    label_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)

    # Generate all the matrices needed for calculations
    gt_boxes, gt_labels = _vectorize_map_gts(gt_instances)
    det_boxes, det_label_prob_mat = _vectorize_map_dets(det_instances)

    # Calculate spatial and label qualities
    label_qual_mat = _calc_label_qual(gt_labels, det_label_prob_mat)
    spatial_qual = _calc_spatial_qual(gt_boxes, det_boxes)

    # Generate the overall cost table (1 - overall quality)
    overall_cost_table[:len(gt_instances), :len(det_instances)] -= _calc_overall_qual(label_qual_mat,
                                                                                      spatial_qual)

    # Generate the spatial and label cost tables
    spatial_cost_table[:len(gt_instances), :len(det_instances)] -= spatial_qual
    label_cost_table[:len(gt_instances), :len(det_instances)] -= label_qual_mat

    return {'overall': overall_cost_table, 'spatial': spatial_cost_table, 'label': label_cost_table}


def _calc_qual_map(gt_instances, det_instances):
    """
    Calculates the sum of qualities for the best matches between ground truth objects and detections for a map.
    Each ground truth object can only be matched to a single detection and vice versa as an object-detection pair.
    Note that if a ground truth object or detection does not have a match, the quality is counted as zero.
    This represents a theoretical object-detection pair with the object or detection and a counterpart which
    does not describe it at all.
    Any provided detection with a zero-quality match will be counted as a false positive (FP).
    Any ground-truth object with a zero-quality match will be counted as a false negative (FN).
    All other matches are counted as "true positives" (TP)
    If there are no ground-truth objects or detections for the map, the system returns zero and this map
    will not contribute to average score.
    :param gt_instances: list of ground-truth dictionaries describing the ground truth objects in the current map.
    :param det_instances: list of detection dictionaries describing the detections for the current map.
    :return: results dictionary containing total overall spatial quality, total spatial quality on positively assigned
    detections, total label quality on positively assigned detections, number of true positives,
    number of false positives, number false negatives, detection evaluation summary,
    and ground-truth evaluation summary for for the given map.
    Format {'overall':<tot_overall_quality>, 'spatial': <tot_tp_spatial_quality>, 'label': <tot_tp_label_quality>,
    'TP': <num_true_positives>, 'FP': <num_false_positives>, 'FN': <num_false_positives>,
    'map_det_evals':<detection_evaluation_summary>, 'map_gt_evals':<ground-truth_evaluation_summary>}
    """

    # Record the full evaluation details for every match
    map_det_evals = []
    map_gt_evals = []
    tot_fp_cost = 0.0

    # if there are no detections or gt instances respectively the quality is zero
    if len(gt_instances) == 0 or len(det_instances) == 0:
        if len(det_instances) > 0:
            # Calculate FP quality
            # TODO handle background class if present and where it is. Currently no background class considered
            tot_fp_cost = np.sum([np.max(det_instance['prob_dist']) for det_instance in det_instances])

            map_det_evals = [{"det_id": idx, "gt_id": None, "ignore": False, "matched": False,
                              "overall": 0.0, "spatial": 0.0, "label": 0.0, "correct_class": None}
                             for idx in range(len(det_instances))]

        elif len(gt_instances) > 0:
            map_gt_evals = [{"det_id": None, "gt_id": idx, "ignore": False, "matched": False,
                             "overall": 0.0, "spatial": 0.0, "label": 0.0, "correct_class": gt_instance.class_label}
                            for idx, gt_instance in enumerate(gt_instances)]

        return {'overall': 0.0, 'spatial': 0.0, 'label': 0.0, 'fp_cost': tot_fp_cost, 'TP': 0, 'FP': len(det_instances),
                'FN': len(gt_instances), "map_det_evals": map_det_evals, "map_gt_evals": map_gt_evals}

    # For each possible pairing, calculate the quality of that pairing and convert it to a cost
    # to enable use of the Hungarian algorithm.
    cost_tables = _gen_cost_tables(gt_instances, det_instances)

    # Use the Hungarian algorithm with the cost table to find the best match between ground truth
    # object and detection (lowest overall cost representing highest overall pairwise quality)
    row_idxs, col_idxs = linear_sum_assignment(cost_tables['overall'])

    # Transform the loss tables back into quality tables with values between 0 and 1
    overall_quality_table = 1 - cost_tables['overall']
    spatial_quality_table = 1 - cost_tables['spatial']
    label_quality_table = 1 - cost_tables['label']

    # Go through all optimal assignments and summarize all pairwise statistics
    # Calculate the number of TPs, FPs, and FNs for the image during the process
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    false_positive_idxs = []

    for match_idx, match in enumerate(zip(row_idxs, col_idxs)):
        row_id, col_id = match
        det_eval_dict = {"det_id": int(col_id), "gt_id": int(row_id), "matched": True, "ignore": False,
                         "overall": float(overall_quality_table[row_id, col_id]),
                         "spatial": float(spatial_quality_table[row_id, col_id]),
                         "label": float(label_quality_table[row_id, col_id]),
                         "correct_class": None}
        gt_eval_dict = det_eval_dict.copy()
        # Handle "true positives"
        if overall_quality_table[row_id, col_id] > 0:
            det_eval_dict["correct_class"] = gt_instances[row_id]["class_id"]
            gt_eval_dict["correct_class"] = gt_instances[row_id]["class_id"]
            true_positives += 1
            map_det_evals.append(det_eval_dict)
            map_gt_evals.append(gt_eval_dict)
        else:
            # Handle false negatives
            if row_id < len(gt_instances):
                gt_eval_dict["correct_class"] = gt_instances[row_id]["class_id"]
                gt_eval_dict["det_id"] = None
                gt_eval_dict["matched"] = False
                false_negatives += 1
                map_gt_evals.append(gt_eval_dict)

            # Handle false positives
            if col_id < len(det_instances):
                det_eval_dict["gt_id"] = None
                det_eval_dict["matched"] = False
                false_positives += 1
                false_positive_idxs.append(col_id)
                map_det_evals.append(det_eval_dict)

    # Calculate the sum of quality at the best matching pairs to calculate total qualities for the image
    tot_overall_img_quality = np.sum(overall_quality_table[row_idxs, col_idxs])

    # Force spatial and label qualities to zero for total calculations as there is no actual association between
    # detections and therefore no TP when this is the case.
    spatial_quality_table[overall_quality_table == 0] = 0.0
    label_quality_table[overall_quality_table == 0] = 0.0

    # Calculate the sum of spatial and label qualities only for TP samples
    tot_tp_spatial_quality = np.sum(spatial_quality_table[row_idxs, col_idxs])
    tot_tp_label_quality = np.sum(label_quality_table[row_idxs, col_idxs])

    # Sort the evaluation details to match the order of the detections and ground truths
    img_det_eval_idxs = [det_eval_dict["det_id"] for det_eval_dict in map_det_evals]
    img_gt_eval_idxs = [gt_eval_dict["gt_id"] for gt_eval_dict in map_gt_evals]
    map_det_evals = [map_det_evals[idx] for idx in np.argsort(img_det_eval_idxs)]
    map_gt_evals = [map_gt_evals[idx] for idx in np.argsort(img_gt_eval_idxs)]

    # Calculate the penalty for assigning a high label probability to false positives
    # TODO handle background class if present and where it is. Currently no background class considered
    tot_fp_cost = np.sum([np.max(det_instances[i]['prob_dist']) for i in false_positive_idxs])

    return {'overall': tot_overall_img_quality, 'spatial': tot_tp_spatial_quality, 'label': tot_tp_label_quality,
            'fp_cost': tot_fp_cost, 'TP': true_positives, 'FP': false_positives, 'FN': false_negatives,
            'map_gt_evals': map_gt_evals, 'map_det_evals': map_det_evals}

