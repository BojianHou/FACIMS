import numpy as np
from data.dataset import load_tadpole, load_credit, load_drug, load_toy_new
from utils.postprocessing import *

def load_dataset(dataset, seed=42):
    if dataset == 'tadpole':
        return load_tadpole(seed)
    elif dataset == 'credit':
        return load_credit(seed)
    elif dataset == 'drug':
        return load_drug(seed)
    elif dataset == 'toy_new':
        return load_toy_new(seed)


def result_show(y, y_hat, A_test, output_dim):
    accuracy, b_acc, recall = compute_accuracy(y, y_hat, output_dim=output_dim)
    if output_dim <= 2:  # binary class
        DP_score = demographic_parity_binary(y_hat, A_test)
        EO_score = equalized_odds_binary(y, y_hat, A_test)
        suf_gap_avg_score = standard_suf_gap_all_binary(y, y_hat, A_test)
        # print("[Recall] The overall recall is:             {:.4f}".format(recall))
    else: # multiple class
        DP_score = demographic_parity_multiclass(y_hat, A_test)
        EO_score = equalized_odds_multiclass(y, y_hat, A_test)
        suf_gap_avg_score = standard_suf_gap_all_multiclass(y, y_hat, A_test)
        # for idx, rec in enumerate(recall):
        #     print("[Recall] The overall recall for class {}:    {:.4f}".format(idx, rec))

    # print("[Accuracy] The overall accuracy is:         {:.4f}".format(accuracy))
    # print("[Balanced Acc] The overall balanced acc is: {:.4f}".format(b_acc))
    # print("[DP] The overall demographic parity is:     {:.4f}".format(DP_score))
    # print("[EO] The overall equalized odds is:         {:.4f}".format(EO_score))
    # print("[SufGAP] The overall sufficiency gap is:    {:.4f}".format(suf_gap_avg_score))

    return accuracy, b_acc, DP_score, EO_score, suf_gap_avg_score, recall


if __name__ == "__main__":

    num_class = 2
    for dataset in ['drug', 'credit', 'tadpole']: # 'tadpole', 'credit', 'drug'
        print("Processing dataset {}...".format(dataset))
        X_train, X_val, X_test, A_train, \
        A_val, A_test, y_train, y_val, y_test = load_dataset(dataset)
        if dataset == 'drug':
            num_class = 4

        for method in [1, 11, 2, 3, 8]:    # 1, 11, 2, 3, 8
            print("method {}...".format(method))
            for lr_prior in [0.1, 0.01]:   # 0.1 0.01 0.001
                print("learning rate for prior model {}".format(lr_prior))
                for lr_post in [0.4, 0.1, 0.01]:   # 0.4 0.1 0.01
                    print("learning rate for post model {}".format(lr_post))
                    acc_list, b_acc_list, dp_list, eo_list, sg_list, recall_list = [], [], [], [], [], []
                    for seed in [0, 42, 666, 777, 1009]:
                        predict = np.load('./npy/{}/prediction_method_{}_seed_{}_lr_prior_{}_lr_post_{}.npy'
                                          .format(dataset, method, seed, lr_prior, lr_post))
                        acc, b_acc, dp, eo, sg, recall = \
                            result_show(y_test, predict, A_test, num_class)
                        acc_list.append(acc)
                        b_acc_list.append(b_acc)
                        dp_list.append(dp)
                        eo_list.append(eo)
                        sg_list.append(sg)
                        recall_list.append(recall)
                    print('===============dataset {}, method {}, lr_prior {}, lr_post {}=================='
                          .format(dataset, method, lr_prior, lr_post))

                    for i in range(num_class):
                        print('Recall {} Mean±Std {:.4f}±{:.4f}'
                              .format(i, np.mean(np.array(recall_list)[:, i]), np.std(np.array(recall_list)[:, i])))

                    print('ACC      Mean±Std {:.4f}±{:.4f}'.format(np.mean(acc_list), np.std(acc_list)))
                    print('BACC     Mean±Std {:.4f}±{:.4f}'.format(np.mean(b_acc_list), np.std(b_acc_list)))
                    print('DP       Mean±Std {:.4f}±{:.4f}'.format(np.mean(dp_list), np.std(dp_list)))
                    print('EO       Mean±Std {:.4f}±{:.4f}'.format(np.mean(eo_list), np.std(eo_list)))
                    print('SG       Mean±Std {:.4f}±{:.4f}'.format(np.mean(sg_list), np.std(sg_list)))

        for method in [7, 9]:    # 1 2 3 8
            print("method {}...".format(method))

            acc_list, b_acc_list, dp_list, eo_list, sg_list, recall_list = [], [], [], [], [], []
            for seed in [0, 42, 666, 777, 1009]:
                predict = np.load('./npy/{}/prediction_method_{}_seed_{}_lr_0.01.npy'
                                      .format(dataset, method, seed))
                acc, b_acc, dp, eo, sg, recall = \
                    result_show(y_test, predict, A_test, num_class)
                acc_list.append(acc)
                b_acc_list.append(b_acc)
                dp_list.append(dp)
                eo_list.append(eo)
                sg_list.append(sg)
                recall_list.append(recall)
            print('===============dataset {}, method {}, lr 0.01=================='
                  .format(dataset, method))

            for i in range(num_class):
                print('Recall {} Mean±Std {:.4f}±{:.4f}'
                      .format(i, np.mean(np.array(recall_list)[:, i]), np.std(np.array(recall_list)[:, i])))

            print('ACC      Mean±Std {:.4f}±{:.4f}'.format(np.mean(acc_list), np.std(acc_list)))
            print('BACC     Mean±Std {:.4f}±{:.4f}'.format(np.mean(b_acc_list), np.std(b_acc_list)))
            print('DP       Mean±Std {:.4f}±{:.4f}'.format(np.mean(dp_list), np.std(dp_list)))
            print('EO       Mean±Std {:.4f}±{:.4f}'.format(np.mean(eo_list), np.std(eo_list)))
            print('SG       Mean±Std {:.4f}±{:.4f}'.format(np.mean(sg_list), np.std(sg_list)))