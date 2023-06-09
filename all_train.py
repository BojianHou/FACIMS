# coding: utf-8
from __future__ import absolute_import, division#, logger.info_function

import os.path as osp
import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import datetime
from data.dataset import load_data
from data.hypers import CALI_PARAMS
from layers.stochastic_models import get_model
from engine.fair_training import inference, train
from utils.postprocessing import result_show, compute_accuracy
from utils.loggers import TxtLogger, set_logger, set_npy, set_npy_new
from utils.common import seed_setup
import logging
import json
try:
    import wandb
except Exception as e:
    pass


def main(prm):
    time_start = time.time()
    seed_setup(prm.seed)

    print('PI is {}'.format(prm.pi))
    
    # log setting
    log_dir, log_file = set_logger(prm)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = TxtLogger(filename=osp.abspath(osp.join(log_dir, log_file)))

    prm.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # added by Bojian
    # prm.device = 'cpu'
    prm.log_var_init = {"mean": prm.log_var_init_mean, "std": prm.log_var_init_var}

    # params setting
    if prm.params is None:
        prm.params = CALI_PARAMS

    # dataloader
    logger.info("=============== DATA LOADING =============")
    X_train, X_val, X_test, A_train, A_val, A_test, y_train, y_val, y_test = load_data(prm)

    # Data Decrease - mimic small data size situation on toxic dataset
    if prm.dataset in ['toxic', 'bank', 'credit']:  # drug tadpole
        A_train_index = []
        for s in np.unique(A_train):
            A_train_index += np.random.choice(
                np.where(A_train == s)[0],
                size=400,
                replace=(len(np.where(A_train == s)[0]) < 400),
            ).tolist()
        A_train = A_train[A_train_index]
        X_train = X_train[A_train_index]
        y_train = y_train[A_train_index]

    if len(X_train.shape) == 4:
        prm.input_shape = X_train.shape[1] * X_train.shape[-2] * X_train.shape[-1]
    else:
        prm.input_shape = len(X_train[0])
    prm.output_dim = len(np.unique(y_train))
    # prm.output_dim = 2
    prm.num_classes = len(np.unique(y_train))

    prm.is_ERM = False
    prm.is_BERM = False
    prm.no_KL = False
    prm.no_indirect_grad = False
    prm.sharp_strategy = False
    prm.no_val = False

    if prm.method == 1:  # FAMS
        pass
    elif prm.method == 2:  # FAMS with manual logits adjustment
        prm.manual_adjust = True
        prm.is_bilevel = True
        prm.no_val = True
    elif prm.method == 3:  # ours: FAMS with automatic logits adjustment
        prm.is_bilevel = True
    elif prm.method == 4:  # ours without KL in up level
        prm.is_bilevel = True
        prm.no_KL = True
    elif prm.method == 5:  # ours without indirect grad for global model f
        prm.is_bilevel = True
        prm.no_indirect_grad = True
    elif prm.method == 6:  # ours without KL in up level and without indirect grad for global model f
        prm.is_bilevel = True
        prm.no_KL = True
        prm.no_indirect_grad = True
    elif prm.method == 7:  # trivial ERM
        prm.is_ERM = True
    elif prm.method == 8:  # ours with sharp strategy
        prm.is_bilevel = True
        prm.sharp_strategy = True
    elif prm.method == 9:
        prm.is_BERM = True
    elif prm.method == 10: # FAMS with sharp strategy
        prm.sharp_strategy = True
    elif prm.method == 11:  # FAMS with indirect gradient
        prm.is_bilevel = True
        prm.no_val = True

    logger.info(prm)
    logger.info("data={}, method={}, lr_prior={}, lr_post={}, "
                "lambda_up={}, lambda_low={}, rho={}".format(
                        prm.dataset, prm.method, prm.lr_prior, prm.lr_post,
                        prm.lambda_up, prm.lambda_low, prm.rho))

    # train
    # train( prm, prior_model, loss_criterion, X_train, A_train, y_train, X_test, A_test, y_test)
    if not os.path.exists('./models'):  # folder 'models' is to save the model parameters
        os.makedirs('./models')

    model = train(prm,
          X_train, A_train, y_train,
          X_val, A_val, y_val,
          X_test, A_test, y_test)

    # evaluation
    logger.info("=============== Inference Process =============")
    if isinstance(model, tuple):  # model[0] is the prior model, model[1] include all post models
        predict_prior = inference(model[0], X_test, prm)
        if prm.dataset == 'toy_new':
            np.save('./npy/{}/prediction_method_{}_seed_{}_lr_prior_{}_lr_post_{}_pi_{}.npy'.
                    format(prm.dataset, prm.method, prm.seed, prm.lr_prior, prm.lr_post, prm.pi),
                    predict_prior)
        else:
            np.save('./npy/{}/prediction_method_{}_seed_{}_lr_prior_{}_lr_post_{}.npy'.
                    format(prm.dataset, prm.method, prm.seed, prm.lr_prior, prm.lr_post),
                    predict_prior)

        acc, b_acc, dp, eo, sg, recall = result_show(y_test, predict_prior, A_test, prm.output_dim)

        group = np.unique(A_test)
        X_test_list = [X_test[np.where(A_test == g)[0]] for g in group]
        y_test_list = [y_test[np.where(A_test == g)[0]] for g in group]
        predict_post_list = [inference(post_model, X_test_list[idx], prm)
                             for idx, post_model in enumerate(model[1])]
        bacc_list = [compute_accuracy(y_test_list[idx], predict_post, prm.acc_bin, prm.output_dim)[1]
                         for idx, predict_post in enumerate(predict_post_list)]
    else:
        predict = inference(model, X_test, prm)
        if prm.dataset == 'toy_new':
            np.save('./npy/{}/prediction_method_{}_seed_{}_lr_{}_pi_{}.npy'.
                    format(prm.dataset, prm.method, prm.seed, prm.lr, prm.pi),
                    predict)
        else:
            np.save('./npy/{}/prediction_method_{}_seed_{}_lr_{}.npy'.
                    format(prm.dataset, prm.method, prm.seed, prm.lr),
                    predict)

        acc, b_acc, dp, eo, sg, recall = result_show(y_test, predict, A_test, prm.output_dim)

    time_end = time.time()
    time_duration = time_end - time_start

    logger.info("The total time for seed {} is: {}".format(prm.seed,
            str(datetime.timedelta(seconds=time_duration))))
    logger.handlers.clear()

    if isinstance(model, tuple):
        return (acc, b_acc, dp, eo, sg, recall, bacc_list)
    else:
        return (acc, b_acc, dp, eo, sg, recall)


if __name__ == "__main__":

    os.environ["WANDB_MODE"] = "offline"
    parser = argparse.ArgumentParser()
    # ----------------------------------------------------------------------------------------------------
    # BASIC param
    # ----------------------------------------------------------------------------------------------------
    parser.add_argument("--config", type=str, help="config file", default='EXPS/toy_new_template.yml')
    # parser.add_argument("--method", type=str, help="method name", default="ours")
    # ----------------------------------------------------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------------------------------------------------
    parser.add_argument("--dataset", type=str, help="dataset name", default="toy_new")
    parser.add_argument("--sens_attrs", type=str, help="sub dataset name for toxic dataset", default="race")
    parser.add_argument("--N_subtask", type=int, help="subgroups number", default=7)
    # toxic kaggle
    parser.add_argument("--acc_bar", type=float, help="evaluation bar for toxic dataset", default=0.4)
    # amazon
    parser.add_argument("--lower_rate", type=int, help="lower review rate for amazon dataset,0-4", default=3)
    parser.add_argument("--upper_rate", type=int, help="lower review rate for amazon dataset,0-4", default=4)
    # training
    parser.add_argument("--model_name", type=str, help="model name", default="FcNet4")
    parser.add_argument("--training_epoch", type=int, help="total training epoch", default=100)
    parser.add_argument("--batch_size", type=int, help="input size for training", default=50)
    # ----------------------------------------------------------------------------------------------------
    # META param
    # ----------------------------------------------------------------------------------------------------
    parser.add_argument("--max_inner", type=int, help="number of inner loop", default=15)
    parser.add_argument("--max_outer", type=int, help="number of outer loop", default=5)
    parser.add_argument("--lr_prior", type=float, help="learning rate for prior model (0.5-1)", default=0.01)
    parser.add_argument("--lr_post", type=float, help="learning rate for post model", default=0.4)
    parser.add_argument("--lr", type=float, help="learning rate for single level ERM model", default=0.01)
    parser.add_argument("--divergence_type", type=str, help="choose the divergence type 'KL' or 'W_Sqr'", default="W_Sqr")
    parser.add_argument("--kappa_prior", type=float, help="The STD of the 'noise' added to prior while using KL", default=0.01)
    parser.add_argument("--kappa_post", type=float, help="The STD of the 'noise' added to post while using KL", default=1e-3)
    parser.add_argument("--lambda_low", type=float, help="trade-off parameter in lower level", default=0.7)
    parser.add_argument("--lambda_up", type=float, help="trade-off parameter in upper level", default=0.7)
    # ----------------------------------------------------------------------------------------------------
    # Stochastic
    # ----------------------------------------------------------------------------------------------------
    parser.add_argument("--log_var_init_mean", type=float, help="Weights initialization (for Bayesian net) - mean", default=-0.1)
    parser.add_argument("--log_var_init_var", type=float, help="Weights initialization (for Bayesian net) - var", default=0.1)
    parser.add_argument("--eps_std", type=float, help="Bayesian Network Noisy Ratio", default=0.1)
    parser.add_argument("--n_MC", type=int, help="Number of Monte-Carlo iterations", default=5)
    # ----------------------------------------------------------------------------------------------------
    # Post Processing
    # ----------------------------------------------------------------------------------------------------
    parser.add_argument("--acc_bin", type=float, help="accuracy bar while evaluating predict result", default=0.5)
    parser.add_argument("--params", type=dict, help="param dict for suf calibration gap calculation", default=None)
    # ----------------------------------------------------------------------------------------------------
    # Other
    # ----------------------------------------------------------------------------------------------------
    parser.add_argument('--cuda_device', default=2, type=int, help='specifies the index of the cuda device')
    parser.add_argument("--seed", type=int, help="seed", default=0)
    parser.add_argument("--use_wandb", type=bool, help="whether use_wandb", default=False)
    parser.add_argument("--wandb_username", type=str, help="wandb user name", default='UNKNOWN')
    parser.add_argument("--exp_name", type=str, help="outpur dir prefix for log info and result", default="test")
    parser.add_argument("--train_inf_step", type=int, help="Inference Period while training", default=1)
    parser.add_argument("--is_bilevel", type=bool, help="whether use bilevel", default=False)
    parser.add_argument("--manual_adjust", type=bool, help="whether manually adjust label", default=False)
    parser.add_argument("--sharp_strategy", type=bool, help="whether use sharp strategy", default=False)
    parser.add_argument("--rho", type=float, help="hyper parameter for sharp strategy",default=0.05)
    parser.add_argument(
        "--method", type=int,
        help="1: FAMS, 2: FAMS+manual logits adjustment"
             "3: Ours, 8: Ours plus sharp strategy (final ours)"
             
             "4: Ours without KL in upper level"
             "5: Ours without indirect grad for global f"
             "6: Ours without KL in upper level-indirect grad for global f"
             
             "7: trivial ERM, 9: balanced ERM, "
             
             "10: FAMS plus sharp strategy, "
             "11: FAMS with indirect gradient",
        default=9
    )
    parser.add_argument('--pi', type=int, help='the ratio between two groups', default=2)
    parser.add_argument('--output_dim', type=int, help='the dimension of output', default=2)

    args = parser.parse_args()
    # torch.cuda.set_device(args.cuda_device)  # set cuda device
    # ----------------------------------------------------------------------------------------------------
    # config file update
    # ----------------------------------------------------------------------------------------------------
    if args.config:
        cfg_dir = osp.abspath(osp.join(osp.dirname(__file__), args.config))
        opt = vars(args)
        args = yaml.load(open(cfg_dir), Loader=yaml.FullLoader)
        opt.update(args)
        args = argparse.Namespace(**opt)

    time_start = time.time()
    logger = logging.getLogger("fair")

    acc_list, b_acc_list, dp_list, eo_list, sg_list, recall_list = [], [], [], [], [], []
    b_acc_post_list = []
    seed_list = [0, 42, 666, 777, 1009]   # 42, 666, 777, 1009
    for seed in seed_list:
        logger.info('==============================seed {}=================================='.format(seed))
        args.seed = seed
        result = main(args)
        if len(result) == 6:
            acc, b_acc, dp, eo, sg, recall = result
        else:
            acc, b_acc, dp, eo, sg, recall, b_acc_post = result
            b_acc_post_list.append(b_acc_post)
        acc_list.append(acc)
        b_acc_list.append(b_acc)
        dp_list.append(dp)
        eo_list.append(eo)
        sg_list.append(sg)
        recall_list.append(recall)

    logger.info('===========================All Results===========================')

    for i in range(args.output_dim):
        logger.info('Recall {}  Mean±Std {:.4f}±{:.4f}'.format(i, np.mean(np.array(recall_list)[:, i]), np.std(np.array(recall_list)[:, i])))
    logger.info('ACC       Mean±Std {:.4f}±{:.4f}'.format(np.mean(acc_list), np.std(acc_list)))
    logger.info('BACC      Mean±Std {:.4f}±{:.4f}'.format(np.mean(b_acc_list), np.std(b_acc_list)))

    logger.info('DP        Mean±Std {:.4f}±{:.4f}'.format(np.mean(dp_list), np.std(dp_list)))
    logger.info('EO        Mean±Std {:.4f}±{:.4f}'.format(np.mean(eo_list), np.std(eo_list)))
    logger.info('SG        Mean±Std {:.4f}±{:.4f}'.format(np.mean(sg_list), np.std(sg_list)))

    if len(b_acc_post_list) > 0:
        b_acc_post_list = np.array(b_acc_post_list)
        mean_list = np.mean(b_acc_post_list, axis=0)
        std_list = np.std(b_acc_post_list, axis=0)
        for idx, mean in enumerate(mean_list):
            logger.info('BACC of post model {} Mean±Std {:.4f}±{:.4f}'.format(idx, mean, std_list[idx]))

    # save result
    npy_dir, npy_file_pre = set_npy_new(args)
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)

    result = {}
    result['acc_list'] = acc_list
    result['b_acc_list'] = b_acc_list
    result['dp_list'] = [float(item) for item in dp_list]
    result['eo_list'] = eo_list
    result['sg_list'] = sg_list
    result['recall_list'] = list(recall_list)
    result['b_acc_post_list'] = [list(item) for item in list(b_acc_post_list)]
    with open(osp.join(npy_dir, npy_file_pre + ".json"), 'w') as file:
        json.dump(result, file)

    # np.save(osp.join(npy_dir, npy_file_pre + "_acc.npy"), acc_list)
    # np.save(osp.join(npy_dir, npy_file_pre + "_bacc_prior.npy"), b_acc_list)
    # np.save(osp.join(npy_dir, npy_file_pre + "_dp.npy"), dp_list)
    # np.save(osp.join(npy_dir, npy_file_pre + "_eo.npy"), eo_list)
    # np.save(osp.join(npy_dir, npy_file_pre + "_sg.npy"), sg_list)
    # if len(b_acc_post_list) > 0:
    #     np.save(osp.join(npy_dir, npy_file_pre + "_bacc_post.npy"), b_acc_post_list)

    time_end = time.time()
    time_duration = time_end - time_start

    logger.info("The total time of {} repeats is: {}".format(len(seed_list),
        str(datetime.timedelta(seconds=time_duration))))
    logger.info("data={}, method={}, lr_prior={}, lr_post={}, lr={}, model={}, lambda_up={}, "
                "lambda_low={}, rho={}".format(args.dataset, args.method, args.lr_prior,
                args.lr_post, args.lr, args.model_name, args.lambda_up, args.lambda_low, args.rho))

