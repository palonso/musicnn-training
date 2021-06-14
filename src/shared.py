import warnings
from ast import literal_eval
from datetime import datetime

import numpy as np
from sklearn import metrics
from sklearn.utils.multiclass import type_of_target

MULTICLASS_INDICATOR = "multiclass-indicator"

warnings.filterwarnings("ignore")


def get_epoch_time():
    return int((datetime.now() - datetime(1970, 1, 1)).total_seconds())


def count_params(trainable_variables):
    # to return number of trainable variables. Example: shared.count_params(tf.trainable_variables()))
    return np.sum([np.prod(v.get_shape().as_list()) for v in trainable_variables])


def load_id2gt(gt_file):
    ids = []
    fgt = open(gt_file)
    id2gt = dict()
    for line in fgt.readlines():
        id, gt = line.strip().split("\t")  # id is string
        id2gt[id] = literal_eval(gt)  # gt is array
        ids.append(id)
    return ids, id2gt


def load_id2path(index_file):
    paths = []
    fspec = open(index_file)
    id2path = dict()
    for line in fspec.readlines():
        id, path = line.strip().split("\t")
        id2path[id] = path
        paths.append(path)
    return paths, id2path


def type_of_groundtruth(y):
    """
    Get the type of groundtruth data by extending scikit learn functionality.

    scikit-learn will detect one-hot encoded multiclass data as multilabl-indicator.
    If this is the case this function returns "multiclass-indicator", which is
    currently not used in scikit-learn, and the scikit-learn result otherwise.

    Args:
        y: numpy array with the groundtruth data

    Returns:
        target_type: string
        Either "multiclass-indicator" or the result of
        sklearn.utils.multiclass.type_of_target
    """
    scikit_learn_type = type_of_target(y)
    if (
        scikit_learn_type == "multilabel-indicator"
        and np.count_nonzero(y) == y.shape[0]
    ):
        return MULTICLASS_INDICATOR
    else:
        return scikit_learn_type


def compute_auc(true, estimated):
    """
    Calculate macro PR-AUC and macro ROC-AUC using the default scikit-learn parameters.

    Multiclass data is currently not supported and ROC-AUC & PR AUC will be NaN.
    """
    estimated = np.array(estimated)
    true = np.array(true)

    if type_of_groundtruth(true) == MULTICLASS_INDICATOR:
        pr_auc = np.nan
        # if we move to scikit-learn 0.22 we can calculate a roc_auc_score for
        # multiclass data like this:
        # estimated = estimated.argmax(axis=1)
        # roc_auc = metrics.roc_auc_score(true, estimated, multi_class="ovr")
        roc_auc = np.nan
    else:
        pr_auc = metrics.average_precision_score(true, estimated)
        roc_auc = metrics.roc_auc_score(true, estimated)
    return roc_auc, pr_auc


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def minmax_standarize(x, x_min=-1, x_max=1, headroom=0.1):
    return (x - x_min) / ((x_max + headroom) - (x_min - headroom))


def average_predictions(pred_array, id_array, ids, id2gt=None):
    # averaging probabilities -> one could also do majority voting
    print('Averaging predictions')
    y_pred = []
    y_true = []
    healthy_ids = []
    for id in ids:
        try:
            avg = np.mean(pred_array[np.where(id_array == id)], axis=0)
            if np.isnan(avg).any():
                print('{} skipped because it contains nans'.format(id))
                continue

            if np.isposinf(avg).any():
                print('{} skipped because it contains pos infs'.format(id))
                continue

            if np.isneginf(avg).any():
                print('{} skipped because it contains neg infs'.format(id))
                continue
            y_pred.append(avg)
            if id2gt:
                y_true.append(id2gt[id])
                healthy_ids.append(id)
        except:
            print(id)

    if id2gt:
        return y_true, y_pred, healthy_ids
    else:
        return y_pred


def average_predictions_ids(pred_array, id_array, ids):
    # averages the predictions and returns the ids of the elements
    # that did not fail.
    print('Averaging predictions')
    y_pred = []
    ids_present = []
    for id in ids:
        try:
            avg = np.mean(pred_array[np.where(id_array == id)], axis=0)
            if np.isnan(avg).any():
                print('{} skipped because it contains nans'.format(id))
                continue

            if np.isposinf(avg).any():
                print('{} skipped because it contains pos infs'.format(id))
                continue

            if np.isneginf(avg).any():
                print('{} skipped because it contains neg infs'.format(id))
                continue
            y_pred.append(avg)
            ids_present.append(id)
        except:
            print(id)

    return y_pred, ids_present


def compute_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print('computing accuracy of {} elements'.format(len(y_true)))

    groundtruth_type = type_of_groundtruth(y_true)
    if groundtruth_type == "multilabel-indicator":
        y_pred = np.round(y_pred)
        return metrics.accuracy_score(y_true, y_pred)
    elif groundtruth_type == MULTICLASS_INDICATOR:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_true = np.squeeze(y_true)
        y_pred = np.round(np.squeeze(y_pred))

    return metrics.balanced_accuracy_score(y_true, y_pred)
