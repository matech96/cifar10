import comet_ml
import holoviews as hv
from comet_ml.api import APIExperiments
from typing import Tuple, List, Dict
from collections import defaultdict


def grid_plot_experiments(comet_api: comet_ml.API, experiments: APIExperiments, p1_name: str, p2_name: str,
                          metrics: List[str] = ['train_acc', 'dev_acc', 'test_acc'], parameters: List[str] = [],
                          fig_size: int = 220) -> hv.core.layout.Layout:
    targets_data = _download_data(comet_api, experiments, p1_name, p2_name, metrics, parameters)

    layout = hv.Layout()
    for k, v in targets_data.items():
        layout += hv.HeatMap(v, kdims=[p1_name, p2_name], vdims=k).sort().opts(title=k).redim.range(z=(0, None))
    return layout.opts(fig_size=fig_size, framewise=True)


def _download_data(comet_api: comet_ml.API, experiments: APIExperiments, p1_name: str, p2_name: str, metrics: List[str],
                   parameters: List[str]) \
        -> Dict[str, List[Tuple[float, float, float]]]:
    targets_data = defaultdict(list)
    list2float = lambda l: float(l[0])
    for experiment in experiments:
        p1_value = list2float(comet_api.get_experiment_parameters(experiment.key, p1_name))
        p2_value = list2float(comet_api.get_experiment_parameters(experiment.key, p2_name))
        for parameter in parameters:
            target_data = list2float(comet_api.get_experiment_parameters(experiment.key, parameter))
            targets_data[parameter].append((p1_value, p2_value, target_data))
        for metric in metrics:
            target_data = list2float(comet_api.get_experiment_metrics(experiment.key, metric))
            targets_data[metric].append((p1_value, p2_value, target_data))
    return targets_data
