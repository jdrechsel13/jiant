import json
import os

import torch

import jiant.utils.python.io as py_io
import jiant.proj.main.components.task_sampler as jiant_task_sampler


def write_val_results(val_results_dict, metrics_aggregator, output_dir, verbose=True, type='val'):
    full_results_to_write = {
        "aggregated": jiant_task_sampler.compute_aggregate_major_metrics_from_results_dict(
            metrics_aggregator=metrics_aggregator,
            results_dict=val_results_dict,
        ),
    }
    for task_name, task_results in val_results_dict.items():
        task_results_to_write = {}
        if "loss" in task_results:
            task_results_to_write["loss"] = task_results["loss"]
        if "metrics" in task_results:
            task_results_to_write["metrics"] = task_results["metrics"].to_dict()
        full_results_to_write[task_name] = task_results_to_write

    metrics_str = json.dumps(full_results_to_write, indent=2)
    if verbose:
        print(metrics_str)

    py_io.write_json(data=full_results_to_write, path=os.path.join(output_dir, f"{type}_metrics.json"))


import numpy as np


def jsonify(obj):
    """
    Recursively convert an object to a JSON-compatible format.

    Args:
        obj: The input object to be converted.

    Returns:
        The JSON-compatible version of the input object.
    """
    if isinstance(obj, dict):
        return {key: jsonify(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [jsonify(element) for element in obj]
    elif isinstance(obj, tuple):
        return tuple(jsonify(element) for element in obj)
    elif isinstance(obj, set):
        return list(jsonify(element) for element in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif hasattr(obj, '__dict__'):
        return jsonify(obj.__dict__)
    else:
        return obj


def write_preds(eval_results_dict, path):
    preds_dict = {}
    for task_name, task_results_dict in eval_results_dict.items():
        preds_dict[task_name] = {
            "preds": task_results_dict["preds"],
            "guids": task_results_dict["accumulator"].get_guids(),
        }
    torch.save(preds_dict, path)

    json.dump(jsonify(preds_dict), open(path.replace('.p', '.json'), 'w+'), indent=2)
