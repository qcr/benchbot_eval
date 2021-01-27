import importlib
import json
import os
import re
import sys
import yaml
import zipfile

DUMP_LOCATION = '/tmp/benchbot_eval_validator_dump'

SKIP_KEY = "SKIP"


def load_results(results_filenames):
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
                                print("\tExtracting data from file '%s'" %
                                      f.filename)
                                results[z.filename + ':' +
                                        f.filename] = json.load(zf)
                            except:
                                print("\tSkipping file '%s'" % f.filename)
        else:
            with open(r, 'r') as f:
                results[r] = json.load(f)
    print("\tDone.\n")
    return results
