import comet_ml
from typing import Callable, TypeVar, List

T = TypeVar('T')


class ProjectWrapper:
    def __init__(self, experiment_name: str):
        self.comet_api = comet_ml.API(api_key="cgss7piePhyFPXRw1J2uUEjkQ", rest_api_key="G9ts5ZxTui6k0ruDTtoQpqCkI")
        self.experiments = self.comet_api.get(experiment_name)

    def get_metrics(self, metrics_name: str, class_constructor: Callable[[str], T]) -> List[T]:
        return [class_constructor(self.comet_api.get_experiment_metrics(experiment.key, metrics_name)[0]) for experiment
                in self.experiments]

    def get_parameter(self, parameter_name: str, class_constructor: Callable[[str], T]) -> List[T]:
        return [class_constructor(self.comet_api.get_experiment_parameters(experiment.key, parameter_name)[0]) for
                experiment in self.experiments]
