**NOTE: this software is part of the BenchBot software stack, and not intended to be run in isolation (although it can be installed independently through pip & run on results files if desired). For a working BenchBot system, please install the BenchBot software stack by following the instructions [here](https://github.com/RoboticVisionOrg/benchbot).**

# BenchBot Evaluation
BenchBot Evaluation contains the code used for evaluating the performance of a BenchBot system in two core semantic scene understanding tasks: semantic SLAM, and scene change detection. The easiest way to use this module is through the helper scripts provided with the [BenchBot software stack](https://github.com/RoboticVisionOrg/benchbot).

For both types of semantic scene understanding task, the tested system will need to produce a results file describing the 3D bounding box location of objects from within the following class list:
```
[bottle, cup, knife, bowl, wine glass, fork, spoon, banana, apple, orange, cake, potted plant, mouse, keyboard, laptop, cell phone, book, clock, chair, table, couch, bed, toilet, tv, microwave, toaster, refrigerator, oven, sink, person]
```

## Installing & performing evaluation with BenchBot Evaluation

BenchBot Evaluation is a Python package, installable with pip. Run the following in the root directory of where this repository was cloned:

```
u@pc:~$ pip install .
```

Although evaluation is best run from within the BenchBot software stack, it can be run in isolation if desired. The following code snippet shows how to perform evaluation from Python:

```python
from benchbot_eval.evaluator import Evaluator

results_filename = "/path/to/your/detection/results.json"
ground_truth_folder = "/path/to/your/ground_truth/folder"
save_file = "/path/to/save/file/for/scores.txt"

my_evaluator = Evaluator(results_filename, ground_truth_folder, save_file)
my_evaluator.evaluate()
```

This prints the final scores to the screen and saves them to the named file:
- `results_filename`: points to the JSON file with the output from your experiment (in the format [described below](#detection-result-format))
- `ground_truth_folder`: the directory containing the relevant environment ground truth JSON files
- `save_file`: is where final scores are to be saved

## Detection result format

Detection results must be presented in a JSON file containing a certain named fields. Some slight differences apply for scene change detection tasks, which are described in more detail [further down](#scene-change-detection-format-differences).

An example of the basic detection results format is as follows:
```
{
    "task_details": {
        "type": <test_challenge_type>,
        "control_mode": <test_control_mode>,
        "localisation_mode": <test_localisation_mode>
    },
    "environment_details": {
        "name": <test_environment_name>,
        "numbers": [<test_environment_numbers>]
    },
    "detections": [
        {
            "class": <object_class_name>,
            "confidence": <object_class_confidence>,
            "centroid": [<xc>, <yc>, <zc>],
            "extent": [<xe>, <ye>, <ze>]
        },
        ...
    ]
}
```

An algorithm attempting to solve a semantic scene understanding task only has to fill in the list of `"detections"`; everything else can be pre-populated using the appropriate [BenchBot API methods](https://github.com/RoboticVisionOrg/benchbot_api). All detections must be given as a dictionary with the following named fields:
- `"class"`: a string which must match a class in the class list at the top of this page
- `"confidence"`: a normalised value between 0 & 1 denoting the system's confidence that the selected class is correct
- `"centroid"`: centre of the detection's 3D bounding box
- `"extent"`: the size dimension of the detection's 3D bounding box along each of the 3 axes (full size, **not** distance from centroid to bounding box edge)

**NOTE: the centroid and extent of the 3D bounding box must be in global coordinates (& metres)**

The `"task_details"` and `"environment_details"` fields define what task was being solved when the results file was produced, & what environment the agent was operating in. These values are defined in the BenchBot software stack when `benchbot_run` is executed to start a new task. For example:

```
u@pc:~$ benchbot_run --task semantic_slam:passive:ground_truth --env miniroom:1
```

would produce the following:

```json
"task_details": {
    "type": "semantic_slam",
    "control_mode": "passive",
    "localisation_mode": "ground_truth"
},
"environment_details": {
    "name": "miniroom",
    "numbers": [1]
}
```

Values for these fields are easily obtained at runtime through the `BenchBot.task_details` & `BenchBot.environment_details` API properties (see [here](https://github.com/RoboticVisionOrg/benchbot_api) for more details). Alternatively, `BenchBot.empty_results()` can be called to create a results `dict` with all fields populated except `"detections"`.

### Format differences for scene change detection tasks

There are two minor differences in results produced for scene change detection tasks:
1) `"environment_details"` contains two numbers referring to the first and second scenes of the environment that will be traversed. For example:

  ```
  u@pc:~$ benchbot_run --task scd:active:ground_truth --env miniroom:1:5
  ```

  would produce the following:

  ```json
  "task_details": {
      "type": "scd",
      "control_mode": "active",
      "localisation_mode": "ground_truth"
  },
  "environment_details": {
      "name": "miniroom",
      "numbers": [1,5]
  }
  ```

2) all detections require an extra `"changed_state"` value, meaning detections must have the following format:
  ```
  ...
  {
      "class": <object_class_name>,
      "confidence": <object_class_confidence>,
      "changed_state": <object_env2_state>,
      "centroid": [<xc>, <yc>, <zc>],
       "extent": [<xe>, <ye>, <ze>]
  }
  ...
  ```
  where `<object_env2_state>` is either the string `"added"` or `"removed"` denoting whether the system suggests the object has been added or removed between the first and second environments.

## Details of evaluation processes

### Semantic SLAM

![semantic_slam_object_map](./docs/semantic_slam_obmap.png)

Solutions to semantic SLAM tasks require the generation of a semantic object map for a given environment that the agent has explored. All object instances from any of the 30 classes must be identified, & 3D bounding boxes must be axis aligned to the world coordinate system. World coordinate frame information is given by `'poses'` observations generated by the BenchBot API. 

Evaluation compares the semantic object map provided by the detections with the ground-truth map of the environment. The evaluation process currently uses 3D mAP and a bird's eye view 2D mAP ("mapbev") to determine the level of success in the task.

### Scene change detection (SCD)

![scene_change_detection_object_map](./docs/scd_obmap.png)

Solutions to scene change detection tasks require the generation of a semantic object map identifying all changes found between two distinct traversals of an environment. Changes to be detected are the removal of some objects, and addition of others, between environment traversals.

As with semantic SLAM tasks, results must report the class label and 3D bounding box location of all changed objects relative to the world coordinate system. A value for the `"changed_state"` field must also be provided declaring if the object has been removed or added to the environment.

Evaluation compares the semantic map of `"added"` and `"removed"` objects with a ground-truth difference map. The library dynamically generates the ground-truth difference map by comparing the ground-truth object maps of the two traversed environments. The evaluation process currently uses the weighted average of mAP scores between the `"added"` and `"removed"` maps formed by separating the appropriate objects from the main semantic object map.
