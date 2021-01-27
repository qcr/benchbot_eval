from benchbot_addons import manager as bam

from . import helpers


class GroundTruthCreator:
    def __init__(self, ground_truth_format_name, environment_name):
        self.ground_truth_format = bam.get_match(
            "formats", [("name", ground_truth_format_name)], return_data=True)
        self.environment = bam.get_match(
            "environments", [("name", bam.env_name(environment_name)),
                             ("variant", bam.env_variant(environment_name))],
            return_data=True)

        self._functions = bam.load_functions(self.ground_truth_format)

    def create_empty(self, *args, **kwargs):
        return {
            'environment':
            self.environment,
            'format':
            self.ground_truth_format,
            'ground_truth': (self._functions['create'](*args, **kwargs)
                             if 'create' in self._functions else {})
        }

    def function(self, name):
        return self._functions[name] if name in self._functions else None
