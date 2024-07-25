import os

import jiant.utils.python.io as py_io
import jiant_v1.proj.simple.runscript as simple_run
import jiant_v1.scripts.download_data.runscript as downloader

import os
import json
import numpy as np



tasks = 'wic,wsc,multirc,rte,copa,record,boolq,cb,superglue_winogender_diagnostics,superglue_broadcoverage_diagnostics'
tasks = 'rte'
tasks = 'record'
tasks='superglue_winogender_diagnostics'
tasks='superglue_broadcoverage_diagnostics'
tasks = 'record' # takes long!!
#tasks = 'wsc'
tasks = 'multirc'


# tasks with expected results
# wic
# multirc (10 epochs!) # todo run for modified models
# copa good for bert-large-cased (high variance, average of 60, benchmark 70), like naive baseline for bert-base-uncased!
# boolq ~72 vs. 77 # todo run for modified models
#
#


# bad
# wsc (still pretty bad!!) 56 vs 64
# cb (f13 almost always 0)


# not working
# diagnostics as only test data available?



# initial rating

# somwhow working
# wsc (overfitted test results)
# cb (test results almost 0)
# wsc (still pretty bad!!) 56 vs 64


# bad: copa (naive baseline)
# boolq worse on val, 0.99 on test????



def run(seed=-1):
    downloader.download_data(tasks.split(','), f"data")

    args = simple_run.RunConfiguration(
        run_name='bert-base-uncased-baseline-' + tasks,
        exp_dir='results/jiant',
        data_dir='data',
        hf_pretrained_model_name_or_path='bert-base-cased',
        tasks=tasks,
        write_test_preds=True,

        #do_test=True,
        #train_batch_size=16,
        num_train_epochs=1,
        learning_rate=1e-5,
        seed=seed,
    )
    ret = simple_run.run_simple(args)

    print('returned')
    print(ret)


def get_model_id(model):
    if not os.path.exists(model):
        # model name is a HF hub model, like bert-base-cased
        model_name = model
    else:
        # model name is the last part of the path
        model_name = model.replace('\\', '/').split('/')[-1]
    return model_name

def average_runs(tasks, n, model='bert-large-cased', output_dir='results', num_train_epochs=10, learning_rate=1e-5):
    """
    Run experiments with different seeds and average the results.

    Parameters:
    - tasks: str, the task(s) to run (e.g., 'wic')
    - n: int, number of runs
    - output_dir: str, directory to save results
    - num_train_epochs: int, number of training epochs
    - learning_rate: float, learning rate

    Returns:
    - averaged_results: dict, mean and std of the results for each task
    """
    # Download data for the task
    downloader.download_data(tasks.split(','), f"data")

    seeds = list(range(n))
    results = []

    model_name = get_model_id(model)

    # Run experiments for each seed
    for seed in seeds:
        run_name = f'{model_name}_{tasks}_{seed}_of_{n}'
        args = simple_run.RunConfiguration(
            run_name=run_name,
            exp_dir=output_dir + '/jiant',
            data_dir='data',
            hf_pretrained_model_name_or_path=model,
            tasks=tasks,
            write_test_preds=True,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            seed=seed,  # Set the seed here,
            #load_best_model=False,
        )
        simple_run.run_simple(args)

        # Collect results from the current run
        result_path = os.path.join(output_dir, 'runs', run_name, 'val_metrics.json')
        with open(result_path, "r") as f:
            results.append(json.load(f))

    # Aggregate results
    averaged_results = {}

    for task in tasks.split(','):
        task_results = [result[task]['metrics']['major'] for result in results]

        averaged_results[task] = {
            "mean": np.mean(task_results),
            "std": np.std(task_results),
            "values": task_results,
        }
        metrics = results[0][task]['metrics']['minor'].keys()
        for metric in metrics:
            task_results = [result[task]['metrics']['minor'][metric] for result in results]
            averaged_results[task][metric] = {
                "mean": np.mean(task_results),
                "std": np.std(task_results),
                "values": task_results,
            }

    print(json.dumps(averaged_results, indent=4))
    file = os.path.join(output_dir, f'{model_name}_{n}.json')
    data = None
    try:
        data = json.load(open(file, 'r'))
        data.update(averaged_results)
    except FileNotFoundError:
        data = averaged_results
    finally:
        if data:
            json.dump(data, open(file, 'w'), indent=2)

    return averaged_results


def _get_model_predictions(model, n, task):
    model_id = get_model_id(model)
    results = []
    for i in range(n):
        run_name = f'{model_id}_{task}_{i}_of_{n}'
        result_path = os.path.join('results/jiant', 'runs', run_name, 'test_preds.json')
        with open(result_path, "r") as f:
            results.append(json.load(f)[task]['preds'])
    return results

def analyze_differences(baseline, model, task, n=5):
    baseline_preds = _get_model_predictions(baseline, n, task)
    model_preds = _get_model_predictions(model, n, task)

    # average the predictions
    baseline_preds_mean = np.mean(baseline_preds, axis=0)
    model_preds_mean = np.mean(model_preds, axis=0)

    # identify the entries that differ the most on average
    diff = np.abs(baseline_preds_mean - model_preds_mean)
    indices = np.argsort(diff)[::-1][:10]

    # print the corresponding entries of the dataset
    #dataset = downloader.(task, 'data')

    pass


if __name__ == '__main__':
    tasks = 'superglue_winogender_diagnostics'
    tasks = 'superglue_broadcoverage_diagnostics'
    tasks = 'rte' # todo! why not working?
    tasks = 'record' # todo
    tasks = 'cb'
    epochs=10
    average_runs(tasks=tasks, n=5, model='bert-large-cased', num_train_epochs=epochs)
    #average_runs(tasks=tasks, n=10, model='bert-base-uncased')
    #average_runs(tasks=tasks, n=10, model='../gradient/results/changed_models/bert-base-uncased-male')
    #average_runs(tasks=tasks, n=10, model='../gradient/results/changed_models/bert-base-uncased-female')
    print('DONE')
