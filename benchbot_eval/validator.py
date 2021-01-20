import json
import yaml
import zipfile

FILE_PATH_KEY = '_file_path'


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


class Validator:

    SKIP_KEY = "SKIP"

    def __init__(self,
                 results_filenames,
                 formats_filenames,
                 ground_truths_filenames,
                 required_task=None,
                 required_envs=None):
        self.formats_data = load_yaml_list(formats_filenames)
        self.ground_truth_data = load_yaml_list(ground_truths_filenames)
        self.results_data = load_results(results_filenames)

    def validate_results_data(self):
        print("Validating data in %d results files:" %
              len(self.results_data.keys()))
        for k, v in self.results_data.items():
            print("\t%s ..." % k)
            assert ('task_details' in v and 'name' in v['task_details']
                    and 'results_format' in v['task_details']), (
                        "Results are missing the bare minimum task details "
                        "('name' & 'resuts_format')")
            assert (
                'environment_details' in v
                and len(v['environment_details']) >= 1
                and all('name' in e for e in v['environment_details'])), (
                    "Results are missing the bare minimum environment details "
                    "(at least 1 item with a 'name' field)")
            assert 'results' in v, "Results has no 'results' field"

            print("\t\tPassed.")

        if self.required_task:
            print(
                "Following results will be skipped due to 'required_task=%s':"
                % self.required_task)
            for k, v in self.results_data.items():
                v[SKIP_KEY] = v['task_details']['name'] != self.required_task
                if v[SKIP_KEY]:
                    print("\t%s" % k)

        if self.required_envs:
            print(
                "Following results will be skipped due to 'required_envs=%s':"
                % self.required_envs)

        return True
