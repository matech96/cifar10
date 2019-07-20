import comet_ml
import pandas as pd
from typing import Callable, TypeVar, List

T = TypeVar('T')


class ProjectWrapper:
    def __init__(self, project_name: str):
        self.comet_api = comet_ml.API(api_key="cgss7piePhyFPXRw1J2uUEjkQ", rest_api_key="G9ts5ZxTui6k0ruDTtoQpqCkI")
        self.experiments = self.comet_api.get(project_name)
        self.name = project_name

    def get_metrics(self, metrics_name: str, class_constructor: Callable[[str], T]) -> List[T]:
        return [class_constructor(self.comet_api.get_experiment_metrics(experiment.key, metrics_name)[0]) for experiment
                in self.experiments]

    def get_parameter(self, parameter_name: str, class_constructor: Callable[[str], T]) -> List[T]:
        return [class_constructor(self.comet_api.get_experiment_parameters(experiment.key, parameter_name)[0]) for
                experiment in self.experiments]

    def get_csvs(self) -> List[pd.DataFrame]:
        return [pd.read_csv('results/{}.csv'.format(e.key)) for e in self.experiments]
