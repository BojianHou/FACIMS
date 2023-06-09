import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.calibration import calibration_curve
import logging
from scipy import interpolate
from scipy.special import softmax
from data.hypers import CALI_PARAMS
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix
from sklearn.preprocessing import normalize
import copy

logger = logging.getLogger("fair")

# ----------------------------------------------------------------------------------------------
# Post Processing and Calculation
# ----------------------------------------------------------------------------------------------
# acc
def compute_accuracy(y, y_hat, bar=0.5, output_dim=2):
    if output_dim == 1:
        y_hat = np.where(y_hat > bar, 1, 0)
    else:
        y_hat = y_hat.argmax(1)

    acc = accuracy_score(y, y_hat)
    bacc = balanced_accuracy_score(y, y_hat)
    # if output_dim <= 2:
    #     recall = recall_score(y, y_hat, average='binary')

    # we calculate the recall for each class separately
    recall = []
    for y_i in np.unique(y):
        idx_current_class = np.where(y == y_i)[0]
        current_recall = len(np.where(y_hat[idx_current_class] == y_i)[0]) / len(idx_current_class)
        recall.append(current_recall)
    return acc, bacc, recall


# suf gap
def new_calibration_curve(y, y_hat, params, normalize=False):

    
    y_hat_normalized = (
        (y_hat - y_hat.min()) / (y_hat.max() - y_hat.min()) if normalize else y_hat
    )
    bins = np.linspace(0.0, 1.0 + 1e-8, params["n_bins"] + 1)
    binids = np.digitize(y_hat_normalized, bins) - 1
    # print(binids)

    bin_sums = np.bincount(binids, weights=y_hat_normalized, minlength=len(bins)-1)
    bin_true = np.bincount(binids, weights=y, minlength=len(bins)-1)
    bin_total = np.bincount(binids, minlength=len(bins)-1)

    nonzero = bin_total != 0
    bin_total[~nonzero] = 1
    prob_true = bin_true / bin_total
    prob_pred = bin_sums / bin_total
    if (
        len(prob_pred) != params["n_bins"]
        or len(prob_true) != params["n_bins"]
        or len(bin_total) != params["n_bins"]
    ):
        print(prob_true, prob_pred, bin_total)
    if params["interpolate_kind"]:
        prob_true, prob_pred = calibration_curve(y, y_hat, normalize=normalize, n_bins=params["n_bins"])

    return prob_true, prob_pred


# def standard_suf_gap_all(y, y_hat, A, prm, if_logger=False):
#     num_A = len(np.unique(A))
#     params = prm.params
#     n_bins = params["n_bins"]
#     interpolate_kind = params["interpolate_kind"]
    
#     groups_pred_true = np.zeros((num_A, n_bins))
#     groups_pred_prob = np.zeros((num_A, n_bins))
#     # all_prob_true, all_prob_pred = new_calibration_curve(y, y_hat, params)
#     all_prob_true, all_prob_pred = calibration_curve(y, y_hat, n_bins=n_bins)
#     for i in range(len(np.unique(A))):
#         try:
#             t, p = new_calibration_curve(
#                 y[A == np.unique(A)[i]], y_hat[A == np.unique(A)[i]], params
#             )
#             if params["interpolate_kind"]:
#                 # new_x = np.linspace(0.01, 0.99, n_bins)
#                 new_x = all_prob_pred
#                 f = interpolate.interp1d(
#                             p,
#                             t,
#                             bounds_error=False,
#                             fill_value=(t[0], t[-1]),
#                             kind=interpolate_kind,
#                         )
#                 t = f(new_x)
#                 p = new_x
#             groups_pred_true[i] = t
#             groups_pred_prob[i] = p
#         except:
#             continue
            
#     exp_x_given_a = np.abs(all_prob_true - groups_pred_true).mean(axis=1)
#     if 'toxic' == prm.dataset:
#         exp_x_a = exp_x_given_a[1:].mean()
#         if if_logger:
#             logger.info("[SufGAP] The average sufficiency gap : {}".format(exp_x_a))
#         return exp_x_a
#     else:
#         exp_x_a = exp_x_given_a.mean()
#         if if_logger:
#             logger.info("[SufGAP] The average sufficiency gap : {}".format(exp_x_a))
#         # logger.info("[SufGAP] The average sufficiency gap : {}".format(exp_x_a))
#         return exp_x_a

def standard_suf_gap_all_binary(y, y_hat, A):
    num_A = len(np.unique(A))
    n_bins = 5
    interpolate_kind = 'linear'

    # groups_pred_true = np.zeros((num_A, n_bins))
    # groups_pred_prob = np.zeros((num_A, n_bins))

    y_hat = softmax(y_hat, axis=1)  # added by Bojian
    y_hat = y_hat[:,1]
    all_prob_true, all_prob_pred = calibration_curve(y, y_hat, n_bins=n_bins, pos_label=1)

    if all_prob_true.shape[0] != n_bins:
        logger.info('The range of prediction is not large enough for sufficiency gap computation, shrink the bins!')
    n_bins = all_prob_true.shape[0]
    groups_pred_true = np.zeros((num_A, n_bins))
    groups_pred_prob = np.zeros((num_A, n_bins))

    # if all_prob_true.shape[0] != n_bins:
    #     all_prob_true = np.zeros(n_bins)

    for i in range(len(np.unique(A))):
        try:
            t, p = calibration_curve(
                y[A == np.unique(A)[i]], y_hat[A == np.unique(A)[i]], n_bins=n_bins)
            if interpolate_kind:
                # new_x = np.linspace(0.01, 0.99, n_bins)
                new_x = all_prob_pred
                f = interpolate.interp1d(
                    p,
                    t,
                    bounds_error=False,
                    fill_value=(t[0], t[-1]),
                    kind=interpolate_kind,
                )
                t = f(new_x)
                p = new_x
            groups_pred_true[i] = t
            groups_pred_prob[i] = p
        except:
            continue

    exp_x_given_a = np.abs(all_prob_true - groups_pred_true).mean(axis=1)
    exp_x_a = exp_x_given_a.mean()

    return exp_x_a


def standard_suf_gap_all_class_wise(y, y_hat, A, class_idx=1):
    num_A = len(np.unique(A))
    n_bins = 5
    interpolate_kind = 'linear'

    y_hat = softmax(y_hat, axis=1)  # normalize
    y_hat = y_hat[:,class_idx]  # choose the specific column (class)

    y_temp = copy.copy(y)
    y_temp[np.where(y != class_idx)[0]] = 0  # make all the other classes as 0
    y_temp[np.where(y == class_idx)[0]] = 1  # make the specific class as 1
    y = y_temp

    all_prob_true, all_prob_pred = calibration_curve(y, y_hat, n_bins=n_bins, pos_label=1)

    if all_prob_true.shape[0] != n_bins:
        logger.info('The range of prediction is not large enough for sufficiency gap computation, shrink the bins!')
    n_bins = all_prob_true.shape[0]
    groups_pred_true = np.zeros((num_A, n_bins))
    groups_pred_prob = np.zeros((num_A, n_bins))

    for i in range(len(np.unique(A))):
        try:
            t, p = calibration_curve(
                y[A == np.unique(A)[i]], y_hat[A == np.unique(A)[i]], n_bins=n_bins)
            if interpolate_kind:
                new_x = all_prob_pred
                f = interpolate.interp1d(
                    p,
                    t,
                    bounds_error=False,
                    fill_value=(t[0], t[-1]),
                    kind=interpolate_kind,
                )
                t = f(new_x)
                p = new_x
            groups_pred_true[i] = t
            groups_pred_prob[i] = p
        except:
            continue

    exp_x_given_a = np.abs(all_prob_true - groups_pred_true).mean(axis=1)
    exp_x_a = exp_x_given_a.mean()

    return exp_x_a


def standard_suf_gap_all_multiclass(y, y_hat, A,):
    max_sg = 0
    for y_i in np.unique(y):
        current_sg = standard_suf_gap_all_class_wise(y, y_hat, A, class_idx=y_i)
        if max_sg < current_sg:
            max_sg = current_sg
    return max_sg


def equalized_odds_binary(y, y_hat, sensitive_features):
    # measure the difference between the true positive rates of different groups
    y_hat = y_hat.argmax(1)

    group_true_pos_r = []
    values_of_sensible_feature = np.unique(sensitive_features)

    true_positive = np.sum([1.0 if y_hat[i] == 1 and y[i] == 1
                             else 0.0 for i in range(len(y_hat))])
    all_positive = np.sum([1.0 if y[i] == 1 else 0.0 for i in range(len(y_hat))])
    all_true_pos_r = true_positive / all_positive

    for val in values_of_sensible_feature:
        positive_sensitive = np.sum([1.0 if sensitive_features[i] == val and y[i] == 1 else 0.0
                                     for i in range(len(y_hat))])
        if positive_sensitive > 0:
            true_positive_sensitive = np.sum([1.0 if y_hat[i] == 1 and
                        sensitive_features[i] == val and y[i] == 1
                         else 0.0 for i in range(len(y_hat))])
            eq_tmp = true_positive_sensitive / positive_sensitive  # true positive rate
            group_true_pos_r.append(eq_tmp)

    return np.mean(np.abs(all_true_pos_r - group_true_pos_r))


def equalized_odds_class_wise(y, y_hat, sensitive_features, class_idx=1):
    # measure the difference between the true positive rates of different groups
    y_hat = y_hat.argmax(1)

    group_true_pos_r = []
    values_of_sensible_feature = np.unique(sensitive_features)

    true_positive = np.sum([1.0 if y_hat[i] == class_idx and y[i] == class_idx
                             else 0.0 for i in range(len(y_hat))])
    all_positive = np.sum([1.0 if y[i] == class_idx else 0.0 for i in range(len(y_hat))])
    all_true_pos_r = true_positive / all_positive

    for val in values_of_sensible_feature:
        positive_sensitive = np.sum([1.0 if sensitive_features[i] == val and y[i] == class_idx else 0.0
                                     for i in range(len(y_hat))])
        if positive_sensitive > 0:
            true_positive_sensitive = np.sum([1.0 if y_hat[i] == class_idx and
                        sensitive_features[i] == val and y[i] == class_idx
                         else 0.0 for i in range(len(y_hat))])
            eo_tmp = true_positive_sensitive / positive_sensitive  # true positive rate
            group_true_pos_r.append(eo_tmp)

    return np.mean(np.abs(all_true_pos_r - group_true_pos_r))


def equalized_odds_multiclass(y, y_hat, sensitive_features):
    # get the maximum EO among all the classes
    max_eo = 0
    for y_i in np.unique(y):
        current_eo = equalized_odds_class_wise(y, y_hat, sensitive_features, class_idx=y_i)
        if max_eo < current_eo:
            max_eo = current_eo
    return max_eo


def demographic_parity_multiclass(y_hat, sensitive_features):
    # get the maximum DP among all the classes
    y_hat = softmax(y_hat, axis=1)  # normalize
    all_p = np.mean(y_hat, axis=0)  # the mean prediction for each class
    p_sensible = []
    values_of_sensible_feature = np.unique(sensitive_features)
    for val in values_of_sensible_feature:
        p_sensible.append(y_hat[np.where(sensitive_features == val)[0]].mean(axis=0))

    return np.max(np.abs(all_p - p_sensible).mean(axis=0))


def demographic_parity_binary(y_hat, sensitive_features):
    # measure the difference between the expectation of prediction between groups
    y_hat = y_hat.argmax(1)
    group_pos_r = []
    values_of_sensible_feature = np.unique(sensitive_features)

    all_positive = np.sum([1.0 if y_hat[i] == 1
                        else 0.0 for i in range(len(y_hat))])
    all_pos_r = all_positive / len(y_hat)

    for val in values_of_sensible_feature:
        sensitive = np.sum([1.0 if sensitive_features[i] == val
                            else 0.0 for i in range(len(y_hat))])
        if sensitive > 0:
            positive_sensitive = np.sum([1.0 if sensitive_features[i] == val and y_hat[i] == 1
                                         else 0.0 for i in range(len(y_hat))])
            eq_tmp = positive_sensitive / sensitive  #  positive rate
            group_pos_r.append(eq_tmp)

    return np.mean(np.abs(all_pos_r - group_pos_r))


# all
def result_show(y, y_hat, A_test, output_dim):
    accuracy, b_acc, recall = compute_accuracy(y, y_hat, output_dim=output_dim)
    if output_dim <= 2:  # binary class
        DP_score = demographic_parity_binary(y_hat, A_test)
        EO_score = equalized_odds_binary(y, y_hat, A_test)
        suf_gap_avg_score = standard_suf_gap_all_binary(y, y_hat, A_test)
    else: # multiple class
        DP_score = demographic_parity_multiclass(y_hat, A_test)
        EO_score = equalized_odds_multiclass(y, y_hat, A_test)
        suf_gap_avg_score = standard_suf_gap_all_multiclass(y, y_hat, A_test)

    for idx, rec in enumerate(recall):
        logger.info("[Recall] The overall recall for class {}:    {:.4f}".format(idx, rec))
    logger.info("[Accuracy] The overall accuracy is:         {:.4f}".format(accuracy))
    logger.info("[Balanced Acc] The overall balanced acc is: {:.4f}".format(b_acc))
    logger.info("[DP] The overall demographic parity is:     {:.4f}".format(DP_score))
    logger.info("[EO] The overall equalized odds is:         {:.4f}".format(EO_score))
    logger.info("[SufGAP] The overall sufficiency gap is:    {:.4f}".format(suf_gap_avg_score))

    return accuracy, b_acc, DP_score, EO_score, suf_gap_avg_score, recall


def result_wandb(y, y_hat, A_test, output_dim=2):
    accuracy, b_acc, recall = compute_accuracy(y, y_hat, output_dim=output_dim)
    if output_dim <= 2:
        suf_gap_avg_score = standard_suf_gap_all_binary(y, y_hat, A_test)
    else:
        suf_gap_avg_score = standard_suf_gap_all_multiclass(y, y_hat, A_test)

    wandb_dict = {
        "acc": accuracy,
        "b_acc": b_acc,
        "recall": recall,
        "suf_gap_avg": suf_gap_avg_score,
    }
    return wandb_dict


def gather_results():

    return
