import pickle
import textwrap

from . import helpers


class Validator:
    def __init__(self,
                 results_filenames,
                 formats_filenames,
                 ground_truths_filenames,
                 required_task=None,
                 required_envs=None):
        self.formats_data = helpers.load_yaml_list(formats_filenames)
        self.ground_truth_data = helpers.load_yaml_list(
            ground_truths_filenames)
        self.results_data = helpers.load_results(results_filenames)

        self.required_task = required_task
        self.required_envs = required_envs

        self.validate_results_data()
        self.dump()

    def _validate_result(self, result_data):
        # Attempt to load the results format
        format_data = next(
            (f for f in self.formats_data
             if f['name'] == result_data['task_details']['results_format']),
            None)
        assert format_data, textwrap.fill(
            "Results declare their format as '%s', "
            "but this format isn't installed in your BenchBot installation" %
            result_data['task_details']['results_format'], 80)

        # Call the validation function if it exists
        fns = helpers.load_functions(format_data)
        if not fns or 'validate' not in fns:
            print("\t\tWARNING: skipping format validation "
                  "('%s' has no validate fn)" % format_data['name'])
            return
        fns['validate'](result_data['results'])

    def dump(self):
        with open(helpers.DUMP_LOCATION, 'wb') as f:
            pickle.dump(self, f)

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
            assert all(
                e['name'] == v['environment_details'][0]['name']
                for e in v['environment_details']), (
                    "Results have multiple environments, rather than multiple "
                    "variants of a single environment")
            assert 'results' in v, "Results has no 'results' field"
            self._validate_result(v)
            print("\t\tPassed.")
            v[helpers.SKIP_KEY] = False

        if self.required_task:
            print(
                "\nFollowing results will be skipped due to 'required_task=%s':"
                % self.required_task)
            for k, v in self.results_data.items():
                v[helpers.SKIP_KEY] = (v['task_details']['name'] !=
                                       self.required_task)
                if v[helpers.SKIP_KEY]:
                    print("\t%s" % k)

        if self.required_envs:
            print("\n%s\n%s" %
                  ("Following results will be skipped due to",
                   textwrap.fill("'required_envs=%s':" % self.required_envs,
                                 80)))
            env_strings = {
                k: helpers.env_string(v['environment_details'])
                for k, v in self.results_data.items()
            }
            skipped = [
                k for k, v in self.results_data.items()
                if env_strings[k] not in self.required_envs
            ]
            if skipped:
                print("\t", end="")
                print("\n\t".join("%s (%s)" % (s, env_strings[s])
                                  for s in skipped))
                for s in skipped:
                    self.results_data[s][helpers.SKIP_KEY] = True
            else:
                print("\tNone.")

            missed = [
                e for e in self.required_envs if e not in env_strings.values()
            ]
            if missed:
                print("\n%s" % textwrap.fill(
                    "WARNING the following required environments have "
                    "no results (an empty result will be used instead):", 80))
                print("\t", end="")
                print("\n\t".join(missed))

        tasks = list(
            set([
                r['task_details']['name'] for r in self.results_data.values()
            ]))
        assert len(tasks) == 1, textwrap.fill(
            "Results are not all for the same task, "
            "& therefore can't be evaluated together. "
            "Received results for tasks '%s' " % tasks, 80)
        return True
