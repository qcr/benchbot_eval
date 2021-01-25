import importlib
import json
import os
import re
import sys
import yaml
import zipfile

DUMP_LOCATION = '/tmp/benchbot_eval_validator_dump'

FILE_PATH_KEY = '_file_path'
SKIP_KEY = "SKIP"


def env_string(envs_data):
    return "%s:%s" % (envs_data[0]['name'], ":".join(
        str(e['variant']) for e in envs_data))


def load_functions(data):
    if 'functions' not in data:
        return {}
    sys.path.insert(0, os.path.dirname(data[FILE_PATH_KEY]))
    ret = {
        k: getattr(importlib.import_module(re.sub('\.[^\.]*$', "", v)),
                   re.sub('^.*\.', "", v))
        for k, v in data['functions'].items()
    }
    del sys.path[0]
    return ret


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


def load_yaml_list(filenames_list):
    return [{
        **{
            FILE_PATH_KEY: f
        },
        **yaml.safe_load(open(f, 'r'))
    } for f in filenames_list]
