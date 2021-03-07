from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from sklearn import metrics


# for decompensation, in-hospital mortality

def print_metrics_binary(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    f1_score=2*prec1*rec1/(prec1+rec1)
    if verbose:
        print("accuracy = {}".format(acc))
        print("precision class 0 = {}".format(prec0))
        print("precision class 1 = {}".format(prec1))
        print("recall class 0 = {}".format(rec0))
        print("recall class 1 = {}".format(rec1))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))
        print("f1_score = {}".format(f1_score))

    return {"acc": acc,
            "prec0": prec0,
            "prec1": prec1,
            "rec0": rec0,
            "rec1": rec1,
            "auroc": auroc,
            "auprc": auprc,
            "minpse": minpse,
            "f1_score":f1_score}


# for phenotyping

def print_metrics_binary_and_multi(y_true, predictions, y_true_flatten, predictions_flatten, verbose=1):
    y_true = np.array(y_true)
    y_true_flatten = np.array(y_true_flatten)
    predictions = np.array(predictions)
    predictions_flatten = np.array(predictions_flatten)
    if len(predictions_flatten.shape) == 1:
        predictions_flatten = np.stack([1 - predictions_flatten, predictions_flatten]).transpose((1, 0))

    ave_auc_micro = metrics.roc_auc_score(y_true, predictions, average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions, average="macro")

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true_flatten, predictions_flatten[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    y_true = np.array([np.argmax(i) for i in y_true])
    predictions = np.array([np.argmax(i) for i in predictions])
    f1_score_micro = metrics.f1_score(y_true, predictions, average="micro")
    f1_score_macro = metrics.f1_score(y_true, predictions, average="macro")

    batch = y_true.shape[0]
    cnt = 0
    for i in range(batch):
        if y_true[i] == predictions[i]:
            cnt += 1
    acc = cnt / batch

    if verbose:
        print("accuracy = {}".format(acc))
        print("ave_auc_micro = {}".format(ave_auc_micro))
        print("ave_auc_macro = {}".format(ave_auc_macro))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))
        print("f1_score_micro = {}".format(f1_score_micro))
        print("f1_score_macro = {}".format(f1_score_macro))

    return {"acc": acc,
            "ave_auc_micro": ave_auc_micro,
            "ave_auc_macro": ave_auc_macro,
            "auprc": auprc,
            "minpse": minpse,
            "f1_score_micro": f1_score_micro,
            "f1_score_macro": f1_score_macro}

def print_metrics_multilabel(y_true, y_pred, verbose=1):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ave_auc_micro = metrics.roc_auc_score(y_true, y_pred,
                                          average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, y_pred,
                                          average="macro")
    
    y_true = np.array([np.argmax(i) for i in y_true])
    y_pred = np.array([np.argmax(i) for i in y_pred])
    batch_size = y_true.shape[0]
    cnt = 0
    for i in range(batch_size):
        if y_true[i] == y_pred[i]:
            cnt += 1
    acc = cnt / batch_size
    
    ave_f1_micro = metrics.f1_score(y_true, y_pred,
                                          average="micro")
    ave_f1_macro = metrics.f1_score(y_true, y_pred,
                                          average="macro")

    if verbose:
        print("ave_auc_micro = {}".format(ave_auc_micro))
        print("ave_auc_macro = {}".format(ave_auc_macro))
        print("ave_f1_micro = {}".format(ave_f1_micro))
        print("ave_f1_macro = {}".format(ave_f1_macro))
        print("acc = {}".format(acc))

    return {"ave_auc_micro": ave_auc_micro,
            "ave_auc_macro": ave_auc_macro,
            "ave_f1_micro": ave_f1_micro,
            "ave_f1_macro": ave_f1_macro,
            "acc": acc}


# for length of stay

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100


def print_metrics_regression(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    predictions = np.maximum(predictions, 0).flatten()
    y_true = np.array(y_true)

    y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
    prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in predictions]
    #print(max(y_true_bins), max(prediction_bins))
    cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    if verbose:
        print("Custom bins confusion matrix:")
        print(cf)

    kappa = metrics.cohen_kappa_score(y_true_bins, prediction_bins,
                                      weights='linear')
    mad = metrics.mean_absolute_error(y_true, predictions)
    mse = metrics.mean_squared_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)

    if verbose:
        print("Mean absolute deviation (MAD) = {}".format(mad))
        print("Mean squared error (MSE) = {}".format(mse))
        print("Mean absolute percentage error (MAPE) = {}".format(mape))
        print("Cohen kappa score = {}".format(kappa))

    return {"mad": mad,
            "mse": mse,
            "mape": mape,
            "kappa": kappa}


class LogBins:
    nbins = 10
    means = [0.611848, 2.587614, 6.977417, 16.465430, 37.053745,
             81.816438, 182.303159, 393.334856, 810.964040, 1715.702848]


def get_bin_log(x, nbins, one_hot=False):
    binid = int(np.log(x + 1) / 8.0 * nbins)
    if binid < 0:
        binid = 0
    if binid >= nbins:
        binid = nbins - 1

    if one_hot:
        ret = np.zeros((LogBins.nbins,))
        ret[binid] = 1
        return ret
    return binid


def get_estimate_log(prediction, nbins):
    bin_id = np.argmax(prediction)
    return LogBins.means[bin_id]


def print_metrics_log_bins(y_true, predictions, verbose=1):
    y_true_bins = [get_bin_log(x, LogBins.nbins) for x in y_true]
    prediction_bins = [get_bin_log(x, LogBins.nbins) for x in predictions]
    cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    if verbose:
        print("LogBins confusion matrix:")
        print(cf)
    return print_metrics_regression(y_true, predictions, verbose)


class CustomBins:
    inf = 1e18
    bins = [(-inf, 7), (7, 35), (35, 63), (63, inf)]
    nbins = len(bins)
    means = [11.450379, 35.070846, 59.206531, 83.382723, 107.487817,
             131.579534, 155.643957, 179.660558, 254.306624, 585.325890]


def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0]
        b = CustomBins.bins[i][1]
        if a <= x < b:
            if one_hot:
                ret = np.zeros((CustomBins.nbins,))
                ret[i] = 1
                return ret
            return i
    return None


def get_estimate_custom(prediction, nbins):
    bin_id = np.argmax(prediction)
    assert 0 <= bin_id < nbins
    return CustomBins.means[bin_id]


def print_metrics_custom_bins(y_true, predictions, verbose=1):
    return print_metrics_regression(y_true, predictions, verbose)
