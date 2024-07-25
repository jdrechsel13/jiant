from run import average_runs

if __name__ == '__main__':
    baseline_model_large = 'bert-large-cased'
    baseline_model_small = 'bert-base-uncased'
    tasks = [ 'record'] # 'wic', 'boolq', 'multirc',
    # multirc ~3.5h for n=10 and bert-base-uncased, bert-large-cased ~10h?



    #tasks = ['wic']

    models = [
        '../gradient/results/changed_models/bert-base-uncased-female',
        baseline_model_small,
        '../gradient/results/changed_models/bert-base-uncased-male',
        baseline_model_large,
       # 'bert-base-cased',
       # 'bert-large-uncased',
        #'roberta-base',
        #'roberta-large',
    ]

    n = 10
    errors = []

    for task in tasks:
        for model in models:
            try:
                average_runs(tasks=task, n=n, model=model)
            except NotImplementedError as e:
                errors.append((task, model, e))

    print('DONE')
    if errors:
        print('Errors:')
        print('\n\t'.join(str(e) for e in errors))