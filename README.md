**NOTE: this software is part of the BenchBot Software Stack, and not intended to be run in isolation. For a working BenchBot system, please install the BenchBot Software Stack by following the instructions [here](https://github.com/RoboticVisionOrg/benchbot).**

# BenchBot Evaluation
BenchBot Evaluation contains the code used for evaluating Semantic SLAM and Scene Change Detections as utilised in the ACRV's BenchBot Scene Understanding challenges. The easiest way to use this module is through the helper scripts provided with BenchBot.

For both challenges, the evaluated system will need to be able to determine the 3D bounding box location of 30 different classes of object.

These 30 classes are:
```
[bottle, cup, knife, bowl, wine glass, fork, spoon, banana, apple, orange, cake, potted plant, mouse, keyboard, laptop, cell phone, book, clock, chair, table, couch, bed, toilet, tv, microwave, toaster, refrigerator, oven, sink, person]
```

## Semantic SLAM Evaluation

![semantic_slam_object_map](./docs/semantic_slam_obmap.png)

The Semantic SLAM challenge requires competitors to generate a semantic object map of a given environment that they have explored. 
All instances of any of the 30 classes must be identified and 3D bounding boxes should be axis aligned to the world coordinate system (world pose information given as part of the benchbot framework upon submitting a solution to the challenge).

Evaluation compares the semantic object map provided with the ground-truth map of the environment.
The evaluation process currently uses 3D mAP and a bird's eye view 2D mAP to determine the level of success in the task.

## Scene Change Detection (SCD) Evaluation

![scene_change_detection_object_map](./docs/scd_obmap.png)

The Scene Change Detection challenge requires competitors to generate a semantic object map of all the changes found between two traversals of an environment.
Changes to be detected are that some objects will be removed, and others added to the environment between traversals.

As with the Semantic SLAM challenge, competitors must report the class label and 3D bounding box location of all changed objects within the world coordinate system.
A label should also be provided defining if the object has been removed or added to the environment.

Evaluation compares the semantic map with "added" and "removed" flags with the ground-truth difference between the ground-truth object maps of the two environments examined.
Currently this evaluation is done using the weighted average of mAP scores between the "added" and "removed" maps formed by separating the appropriate objects from the main semantic object map.

## Detection Result Format

Detection results should be presented in a json file with a standard basic format. Some slight differences apply for scene change detection which you can find [here](###scene-change-detection-format-differences).

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
        "numbers": [<test_environment_number>]
    },
    "detections": [
        {
            "class": <object_class_name>,
            "confidence": <object_class_confidence>,
            "centroid": [<xc>, <yc>, <zc>],
            "extent": [<xe>, <ye>, <ze>]
        },
        .
        .
        .
    ]
}
```

The "task_details" and "environment_details" define what problem this results file was trying to solve.
These are defined within benchbot when you perform `benchbot_run` command to start an experiment.
These should be attainable as part of the benchbot_api which you can find [here](https://github.com/RoboticVisionOrg/benchbot_api) but below is a quick breakdown of how it should look.

As an example, the command: 

```
benchbot_run --task semantic_slam:passive:ground_truth --env miniroom:1
```

would equate to the following task and environment details:

```
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

All detections are given as a dictionary defining the object.
The value of "class" is given as a string and must match one in the class list given in the class list at the top of the page.
The "confidence" provided is the confidence the system has that the class given is correct.
The "centroid" and "extent" define the location of the detected object

**Important!** values for the centroid and extent of the detection are given in metres and are in global coordinates. Extent defines the absolute length of the bounding box along the x, y, and z axes and not the distance from the centroid to the bounding box edge.

### Scene Change Detection Format Differences
While mostly following the format given above, some slight differences exist for scene change detection.

Firstly, "environment_details" will contain two numbers for the first and second versions of the environment you traverse. 

For example:
```
benchbot_run --task scd:active:ground_truth --env miniroom:15
```

would equate to the following task and environment details:

```
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

Secondly, all detection dictionaries will have an extra "changed_state" key, leaving all detections with the following format:
```
{
    "class": <object_class_name>,
    "confidence": <object_class_confidence>,
    "changed_state": <object_env2_state>,
    "centroid": [<xc>, <yc>, <zc>],
     "extent": [<xe>, <ye>, <ze>]
}
```
where `<object_env2_state>` is either "added" or "removed" depending on whether the system thinks the object has been added or removed between the first and second environments.

# Package Requirements
As per the note at the top of this page, this repo should not be used in isolation, however the package requirements are as follows:

```
numpy
shapely
```

# Running BenchBot Evaluate
This should not be done in isolation but as part of the BenchBot framework as per the note at the top of this page.
However, using this package in isolation is done as follows in python:

```
from benchbot_eval.evaluator import Evaluator

results_filename = "/path/to/your/detection/results.json"
ground_truth_folder = "/path/to/your/ground_truth/folder"
save_file = "/path/to/save/file/for/scores.txt"

my_evaluator = Evaluator(results_filename, ground_truth_folder, save_file)
my_evaluator.evaluate()
```

where the results_filename should point to the .json file with the output from your experiment in the format described in [Detection Results Format](##detection-results-format), ground_truth_folder should point to the directory containing all ground-truth file .jsons for all environments, and save_file should be where you wish to save your final scores to.

This will print the final scores to the screen and save them to the named file.