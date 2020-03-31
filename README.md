**NOTE: this software is part of the BenchBot software stack, and not intended to be run in isolation (although it can be installed independently through pip & run on results files if desired). For a working BenchBot system, please install the BenchBot software stack by following the instructions [here](https://github.com/RoboticVisionOrg/benchbot).**

# BenchBot Evaluation
BenchBot Evaluation contains the code used for evaluating the performance of a BenchBot system in two core semantic scene understanding tasks: semantic SLAM, and scene change detection. The easiest way to use this module is through the helper scripts provided with the [BenchBot software stack](https://github.com/RoboticVisionOrg/benchbot).

For both types of semantic scene understanding task, the tested system will need to produce a results file describing the 3D bounding box (cuboid) location of objects from within the following class list:
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

results_filename = "/path/to/your/proposed/map/results.json"
ground_truth_folder = "/path/to/your/ground_truth/folder"
save_file = "/path/to/save/file/for/scores.txt"

my_evaluator = Evaluator(results_filename, ground_truth_folder, save_file)
my_evaluator.evaluate()
```

This prints the final scores to the screen and saves them to the named file:
- `results_filename`: points to the JSON file with the output from your experiment (in the format [described below](##results-format))
- `ground_truth_folder`: the directory containing the relevant environment ground truth JSON files
- `save_file`: is where final scores are to be saved

## Results format

Results must be presented in a JSON file containing a certain named fields. Some slight differences apply for scene change detection tasks, which are described in more detail [further down](#scene-change-detection-format-differences).

An example of the basic proposed map results format is as follows:
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
    "proposals": [
        {
            "label_probs": [<object_class_probability_distribution>],
            "centroid": [<xc>, <yc>, <zc>],
            "extent": [<xe>, <ye>, <ze>]
        },
        ...
    ]
    "class_list": [<classes_order>]
}
```

An algorithm attempting to solve a semantic scene understanding task only has to fill in the list of `"proposals"` and (optionally) the `"classes"` field; everything else can be pre-populated using the appropriate [BenchBot API methods](https://github.com/RoboticVisionOrg/benchbot_api).  
All proposed objects for the final map must be given as a dictionary with the following named fields:
- `"label_probs"`: a probability distribution for all classes as ordered in your `"class_list"` field.
- `"centroid"`: centre of the proposed object cuboid.
- `"extent"`: the size dimension of the proposed object cuboid along each of the 3 axes (full size, **not** distance from centroid to cuboid edge)

**NOTE: any probability distributions with a total probability greater than 1 will be re-normalized.**

**NOTE: the centroid and extent of the cuboid must be axis aligned in global coordinates (& metres)**

The `"class_list"` field allows you to define the order that probabilities given in an object proposal's `"label_probs"` field.
This can include fewer objects than in the list above and can include a `"background"` class.
The format is a list of strings composed of classes in the class list given above.
There is some support given for synonyms which you can find in `benchbot_eval/class_list.py`.

**NOTE: any class names given that are not in the class list given (or are an appropriate synonym) will be ignored.**

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

Values for these fields are easily obtained at runtime through the `BenchBot.task_details` & `BenchBot.environment_details` API properties (see [here](https://github.com/RoboticVisionOrg/benchbot_api) for more details). Alternatively, `BenchBot.empty_results()` can be called to create a results `dict` with all fields populated except `"proposals"`.

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

2) all object proposals require an extra `"state_probs"` value, meaning object proposals must have the following format:
  ```
  ...
  {
      "label_probs": [<object_class_probability_distribution>],
      "state_probs": [<pa>, <pr>, <ps>],
      "centroid": [<xc>, <yc>, <zc>],
       "extent": [<xe>, <ye>, <ze>]
  }
  ...
  ```
  where `<pa>`, `<pr>` and `<ps>` are the probabilities the object is in any of the possible states for the SCD task (added, removed, and same respectively).

## Details of evaluation processes

### Semantic SLAM

![semantic_slam_object_map](./docs/semantic_slam_obmap.png)

Solutions to semantic SLAM tasks require the generation of a semantic object map for a given environment that the agent has explored. All object instances from any of the 30 classes must be identified, & object cuboids in the map must be axis aligned to the world coordinate system. World coordinate frame information is given by `'poses'` observations generated by the BenchBot API. 

Evaluation compares the semantic object map provided by the agent with the ground-truth map of the environment using a novel object map quality (OMQ) measure.
This measure is based on the probabilistic object detection quality measure PDQ (PDQ [paper](http://openaccess.thecvf.com/content_WACV_2020/papers/Hall_Probabilistic_Object_Detection_Definition_and_Evaluation_WACV_2020_paper.pdf) and [code](https://github.com/david2611/pdq_evaluation))

#### Object Map Quality (OMQ)
Object map quality compares ground-truth cuboids of a scene with the proposed object cuboids of a generated map.
It compares both how well proposals overlap with ground-truth objects and how well they have semantically labelled the objects in the map.
The steps for calculating this are as follows:

1. For every object in the proposed map, we compare it to all ground-truth objects both spatially and semantically, calculating a quality score for each.

    a) **Spatial Quality** is the 3D IoU of the ground-truth and proposed cuboids being compared.
    
    b) **Label Quality** is the probability given to the correct class label of the ground-truth object being compared to.

    c) The final pairwise object quality for the proposed and ground-truth objects is then the geometric mean of these two sub-qualities.
**Note**, if either quality score is zero (e.g. no overlap between proposed and ground-truth cuboids), the pairwise score will be zero.

2. We optimally assign proposed objects from your map with the ground-truth objects, establishing our "true positives" (some non-zero quality), false negatives (no pairwise quality match ground-truth objects), and false positives (no pairwise quality match proposed objects).

3. We calculate a false positive cost for all false positive proposed objects (maximum confidence given to non-background class)

4. We calculate an overall OMQ score as the sum of all "true positive" qualities divided by the number of "true positives", number of false negatives, and total false positive cost

You will receive as output this overall OMQ score, alongside average pairwise overall, spatial, and label qualities averaged over the number of "true positive" objects.

### Scene change detection (SCD)

![scene_change_detection_object_map](./docs/scd_obmap.png)

Solutions to scene change detection tasks require the generation of a semantic object map identifying all changes found between two distinct traversals of an environment. Changes to be detected are the removal of some objects, and addition of others, between environment traversals.

As with semantic SLAM tasks, results must report the class label and cuboid location of all changed objects relative to the world coordinate system. 

A probability distribution in the `"state_probs"` field must also be provided giving the confidence that an object has been added, removed, or remained the same.

Evaluation is given as a variation of the OMQ measure described [above](####object-map-quality-omq).

Differences in how the SCD-variant are calculated are given below.

#### Object Map Quality - SCD

The only differences between the calculation of OMQ for SCD which differ from semantic slam are in the calculation of overall pairwise quality, and the calculation of false positive cost (steps 1.c and 3. respectively) due to the addition of a new state quality measure.

**State Quality** is calculated the same as label quality in the standard OMQ score.
Specifically, the state quality is equal to the probability given to the correct state (e.g. [0.4, 0.5, 0.1] on an added object would get a state score of 0.4) 

**Final Pairwise Score** (step 1.c in previous instructions) is now calculated as the geometric mean of the three sub-quality scores (spatial, label, and state) rather than of the original two (spatial and label).

**False Positive Cost** (step 3. in previous instructions) is now the geometric mean of both the maximum label probability of a non-background class and the maximum state probability of a non-same class.
This means that both overconfidence in class and in state will contribute to the cost of a false positive object proposal.

**NOTE the inclusion of state quality will effect the pairwise scores of individual proposed and ground-truth objects from what they would have been in Semantic SLAM (averaging over 3 terms instead of 2)**
 
