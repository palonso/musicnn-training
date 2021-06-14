import argparse
import json
from collections import defaultdict, namedtuple
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

import shared

Score = namedtuple('Score', ['mean', 'std'])


def score_predictions(args):
    config = json.load(open(Path(args.config_file), 'r'))

    results_file = Path(config['exp_dir'], 'results_whole')
    predictions_file = Path(config['exp_dir'], 'predictions')

    ids, folds, predictions, groundtruth = [], [], dict(), dict()

    n_folds = config['config_train']['n_folds']
    grondtruth_folder = Path(config['config_train']['gt_test']).parent
    dataset = config['dataset']

    # get fold-wise predictions
    for i in range(n_folds):
        groundtruth_file = grondtruth_folder / f"gt_test_{i}.csv"

        with open('{}_{}.json'.format(predictions_file, i), 'r') as f:
            fold = json.load(f)
            folds.append(fold)
            predictions.update(fold)

        ids_fold, gt_fold = shared.load_id2gt(groundtruth_file)

        ids += ids_fold
        groundtruth.update(gt_fold)

    groundtruth_ids = set(ids)
    predictions_ids = set(predictions.keys())

    # check if there are missing predictions and update ids
    missing_ids = groundtruth_ids.symmetric_difference(predictions_ids)
    if missing_ids:
        print('ids without predictions or groundtruth: {}'.format(missing_ids))
        ids = list(predictions_ids - missing_ids)

    y_true, y_pred = zip(*[(groundtruth[i], predictions[i]) for i in ids])

    fold_gt, fold_pred = [], []
    for i, fold in enumerate(folds):
        keys = [i for i in fold.keys() if i in groundtruth.keys()]
        fold_pred.append([predictions[k] for k in keys])
        fold_gt.append([groundtruth[k] for k in keys])

    roc_auc_score, pr_auc_score, macro_acc_score, micro_acc, report = get_metrics(
        y_true, y_pred, fold_gt, fold_pred, n_folds
    )

    store_results(
        results_file,
        roc_auc_score,
        pr_auc_score,
        macro_acc_score,
        micro_acc,
        report,
        dataset,
    )


def get_metrics(y_true, y_pred, fold_gt, fold_pred, n_folds):
    micro_acc = shared.compute_accuracy(y_true, y_pred)
    accs = []
    roc_aucs, pr_aucs = [], []
    for i in range(n_folds):
        y_true_fold = fold_gt[i]
        y_pred_fold = fold_pred[i]
        accs.append(shared.compute_accuracy(y_true_fold, y_pred_fold))
        roc_auc_i, pr_auc_i = shared.compute_auc(y_true_fold, y_pred_fold)
        roc_aucs.append(roc_auc_i)
        pr_aucs.append(pr_auc_i)

    macro_acc = np.mean(accs)
    print("Macro Acc:", macro_acc)
    print("Micro Acc:", micro_acc)
    acc_std = np.std(accs)
    roc_auc, pr_auc = np.mean(roc_aucs), np.mean(pr_aucs)
    roc_auc_std, pr_auc_std = np.std(roc_aucs), np.std(pr_aucs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if shared.type_of_groundtruth(y_true) == "multilabel-indicator":
        y_pred_indicator = np.round(y_pred)
        report = classification_report(y_true, y_pred_indicator)
    else:
        y_true_argmax = np.argmax(y_true, axis=1)
        y_pred_argmax = np.argmax(y_pred, axis=1)
        report = classification_report(y_true_argmax, y_pred_argmax)

    roc_auc_score = Score(roc_auc, roc_auc_std)
    pr_auc_score = Score(pr_auc, pr_auc_std)
    macro_acc_score = Score(macro_acc, acc_std)
    return roc_auc_score, pr_auc_score, macro_acc_score, micro_acc, report


def store_results(
    output_file,
    roc_auc_score,
    pr_auc_score,
    macro_acc_score,
    micro_acc,
    report,
    dataset,
):
    # print experimental results
    print('ROC-AUC: ' + str(roc_auc_score.mean))
    print('PR-AUC: ' + str(pr_auc_score.mean))
    print('Balanced Micro Acc: ' + str(micro_acc))
    print('Balanced Macro Acc: ' + str(macro_acc_score.mean))
    print('Balanced Acc STD: ' + str(macro_acc_score.std))
    print('latext format:')
    print('{:.2f}\\pm{:.2f}'.format(micro_acc, macro_acc_score.std))
    print('-' * 20)

    # store experimental results
    with open(output_file, 'w') as to:
        to.write('\nROC AUC: ' + str(roc_auc_score.mean))
        to.write('\nStD: ' + str(roc_auc_score.std))
        to.write('\nPR AUC: ' + str(pr_auc_score.mean))
        to.write('\nStD: ' + str(pr_auc_score.std))
        to.write('\nAcc Micro: ' + str(micro_acc))
        to.write('\nAcc Macro: ' + str(macro_acc_score.mean))
        to.write('\nStD: ' + str(macro_acc_score.std))
        to.write('\n')
        to.write('Report:\n')
        to.write('{}\n'.format(report))

    output_summary = output_file.parent.parent / 'results.json'

    try:
        with open(output_summary, 'r') as fp:
            data = json.load(fp)
    except:
        data = dict()

    with open(output_summary, 'w+') as fp:
        data[dataset] = defaultdict(dict)
        data[dataset]['Accuracy Micro']['mean'] = micro_acc
        data[dataset]['Accuracy Macro']['mean'] = macro_acc_score.mean
        data[dataset]['Accuracy']['std'] = macro_acc_score.std
        data[dataset]['ROC AUC']['mean'] = roc_auc_score.mean
        data[dataset]['ROC AUC']['std'] = roc_auc_score.std
        data[dataset]['PR AUC']['mean'] = pr_auc_score.mean
        data[dataset]['PR AUC']['std'] = pr_auc_score.std

        json.dump(data, fp, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='configuration file')
    args = parser.parse_args()

    score_predictions(args)
