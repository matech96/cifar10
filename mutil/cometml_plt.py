import comet_ml
import holoviews as hv
from comet_ml.api import APIExperiments
from typing import Tuple, List


def grid_plot_experiments(comet_api: comet_ml.API, experiments: APIExperiments, p1_name: str, p2_name: str,
                          fig_size: int = 220) -> hv.core.layout.Layout:
    train_data, dev_data, test_data = _download_data(comet_api, experiments, p1_name, p2_name)

    return (hv.HeatMap(train_data, kdims=[p1_name, p2_name]).sort().opts(title="Train") +
            hv.HeatMap(dev_data, kdims=[p1_name, p2_name]).sort().opts(title="Dev") +
            hv.HeatMap(test_data, kdims=[p1_name, p2_name]).sort().opts(title="Test")).opts(fig_size=fig_size)


def _download_data(comet_api: comet_ml.API, experiments: APIExperiments, p1_name: str, p2_name: str) \
        -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    train_data = []
    dev_data = []
    test_data = []
    list2float = lambda l: float(l[0])
    for experiment in experiments:
        train_acc = list2float(comet_api.get_experiment_metrics(experiment.key, 'train_acc'))
        dev_acc = list2float(comet_api.get_experiment_metrics(experiment.key, 'dev_acc'))
        test_acc = list2float(comet_api.get_experiment_metrics(experiment.key, 'test_acc'))
        p1_value = list2float(comet_api.get_experiment_parameters(experiment.key, p1_name))
        p2_value = list2float(comet_api.get_experiment_parameters(experiment.key, p2_name))
        train_data.append((p1_value, p2_value, train_acc))
        dev_data.append((p1_value, p2_value, dev_acc))
        test_data.append((p1_value, p2_value, test_acc))
    return train_data, dev_data, test_data
