**NOTE: this software is part of the BenchBot software stack, and not intended to be run in isolation (although it can be installed independently through pip and run on results files if desired). For a working BenchBot system, please install the BenchBot software stack by following the instructions [here](https://github.com/qcr/benchbot).**

# BenchBot Evaluation

[![BenchBot project](https://img.shields.io/badge/collection-BenchBot-%231a2857)](http://benchbot.org)
[![QUT Centre for Robotics Open Source](https://github.com/qcr/qcr.github.io/raw/master/misc/badge.svg)](https://qcr.github.io)
![Primary language](https://img.shields.io/github/languages/top/qcr/benchbot_eval)
[![License](https://img.shields.io/github/license/qcr/benchbot_eval)](./LICENSE.txt)

BenchBot Evaluation is a library of functions used to call evaluation methods. These methods are installed through the [BenchBot Add-ons Manager](https://github.com/qcr/benchbot-addons), and evaluate the performance of a BenchBot system against the metric. The easiest way to use this module is through the helper scripts provided with the [BenchBot software stack](https://github.com/qcr/benchbot).

## Installing and performing evaluation with BenchBot Evaluation

BenchBot Evaluation is a Python package, installable with pip. Run the following in the root directory of where this repository was cloned:

```
u@pc:~$ pip install .
```

Although evaluation is best run from within the BenchBot software stack, it can be run in isolation if desired. The following code snippet shows how to perform evaluation with the `'omq'` method from Python:

```python
from benchbot_eval.evaluator import Evaluator, Validator

Validator(results_file).validate_results_data()
Evaluator('omq', scores_file).evaluate()
```

This prints the final scores to the screen and saves them to a file using the following inputs:

- `results_file`: points to the JSON file with the output from your experiment
- `ground_truth_folder`: the directory containing the relevant environment ground truth JSON files
- `save_file`: is where final scores are to be saved

## How add-ons interact with BenchBot Evaluation

Two types of add-ons are used in the BenchBot Evaluation process: format definitions, and evaluation methods. An evaluation method's YAML file defines what results formats and ground truth formats the method supports. This means:

- this package requires installation of the [BenchBot Add-ons Manager](https://github.com/qcr/benchbot_addons) for interacting with installed add-ons
- the `results_file` must be a valid instance of a supported format
- there must be a valid ground truth available in a supported format, for the same environment as the results
- validity is determined by the format-specific validation function described in the format's YAML file

Please see the [BenchBot Add-ons Manager's documentation](https://github.com/qcr/benchbot_addons) for further details on the different types of add-ons.

## Creating valid results and ground truth files

The [BenchBot software stack](https://github.com/qcr/benchbot) includes tools to assist in creating results and ground truth files:

- **results:** are best created using the `empty_results()` and `results_functions()` helper functions in the [BenchBot API](https://github.com/qcr/benchbot_api), which automatically populate metadata for your current task and environment.
- **ground truths:** this package includes a `GroundTruthCreator` class to aid in creating ground truths of a specific format, for a specific environment. Example use includes:

  ```python
  from benchbot_eval.ground_truth_creator import GroundTruthCreator

  gtc = GroundTruthCreator('object_map_ground_truth', 'miniroom:1')
  gt = gtc.create_empty();
  print(gtc.functions())  # ['create', 'create_object']
  gt['ground_truth']['objects'][0] = gtc.functions('create_object')
  ```
