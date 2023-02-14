from __future__ import absolute_import, division, print_function

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

from utils.postprocessing import (
    result_show,
)

from utils.loggers import TxtLogger, set_logger, set_npy
from utils.common import seed_setup



try:
    import wandb
except Exception as e:
    pass


def main(prm):
    time_start = time.time()

    seed_setup(prm.seed)
    
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

    
    # Data Decrease for toxic - mimic small data size situation on toxic dataset
    if prm.dataset == "toxic":
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
    elif prm.dataset == 'bank':
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
    elif prm.dataset in ["adult"]:
        A_train_index = []
        for s in np.unique(A_train):
            A_train_index += np.random.choice(
                np.where(A_train == s)[0],
                size=500,
                replace=(len(np.where(A_train == s)[0]) < 500),
            ).tolist()
        A_train = A_train[A_train_index]
        X_train = X_train[A_train_index]
        y_train = y_train[A_train_index]
    elif prm.dataset in ["celeba"]:
        A_train_index = []
        for s in np.unique(A_train):
            A_train_index += np.random.choice(
                np.where(A_train == s)[0],
                size=200,
                replace=(len(np.where(A_train == s)[0]) < 200),
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

    # if we use bilevel optimization to tackle
    # imbalance issue by adjusting logits automatically,
    # we then cannot use manual way to adjust logits
    # if prm.is_bilevel:
    #     prm.manual_adjust = False

    prm.is_ERM = False
    prm.is_BERM = False
    prm.no_KL = False
    prm.no_indirect_grad = False

    if prm.method == 1:  # FAMS
        pass
    elif prm.method == 2:  # FAMS with manual logits adjustment
        prm.manual_adjust = True
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
    elif prm.method == 8:
        prm.is_bilevel = True
        prm.sharp_strategy = True
    elif prm.method == 9:
        prm.is_BERM = True

    logger.info(prm)
    logger.info("data={}, method={}, lr_prior={}, lr_post={}, "
                "lambda_up={}, lambda_low={}, rho={}".format(
                        prm.dataset, prm.method, prm.lr_prior, prm.lr_post,
                        prm.lambda_up, prm.lambda_low, prm.rho))

    # train
    # train( prm, prior_model, loss_criterion, X_train, A_train, y_train, X_test, A_test, y_test)
    model = train(prm,
          X_train, A_train, y_train,
          X_val, A_val, y_val,
          X_test, A_test, y_test)

    # evaluation
    logger.info("=============== Inference Process =============")

    predict = inference(model, X_test, prm)

    # save result
    npy_dir, npy_file_pre = set_npy(prm, prm.training_epoch)
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)

    np.save(osp.join(npy_dir, npy_file_pre + "_testy.npy"), y_test)
    np.save(osp.join(npy_dir, npy_file_pre + "_testA.npy"), A_test)
    np.save(osp.join(npy_dir, npy_file_pre + "_predict.npy"), predict)

    # logger.info("=============== Post Process =============")

    result_show(y_test, predict, A_test, prm)

    time_end = time.time()
    time_duration = time_end - time_start
    logger.info("data={}, method={}, lr_prior={}, lr_post={}, lambda_up={}, "
                "lambda_low={}, rho={}".format(prm.dataset, prm.method, prm.lr_prior,
                                       prm.lr_post, prm.lambda_up, prm.lambda_low, prm.rho))
    logger.info("The total time is: {}".format(
            str(datetime.timedelta(seconds=time_duration))))

    logger.handlers.clear()



if __name__ == "__main__":

    os.environ["WANDB_MODE"] = "offline"
    parser = argparse.ArgumentParser()
    # ----------------------------------------------------------------------------------------------------
    # Direct args
    # ----------------------------------------------------------------------------------------------------
    # BASIC param
    # ----------------------------------------------------------------------------------------------------
    parser.add_argument("--config", type=str, help="config file", default='EXPS/bank_template.yml')

    # parser.add_argument("--method", type=str, help="method name", default="ours")
    # DATASET
    parser.add_argument("--dataset", type=str, help="dataset name", default="tadpole")
    parser.add_argument(
        "--sens_attrs",
        type=str,
        help="sub dataset name for toxic dataset",
        default="race",
    )
    parser.add_argument("--N_subtask", type=int, help="subgroups number", default=7)
    # toxic kaggle
    parser.add_argument(
        "--acc_bar", type=float, help="evaluation bar for toxic dataset", default=0.4
    )
    # amazon
    parser.add_argument(
        "--lower_rate",
        type=int,
        help="lower review rate for amazon dataset,0-4",
        default=3,
    )
    parser.add_argument(
        "--upper_rate",
        type=int,
        help="lower review rate for amazon dataset,0-4",
        default=4,
    )

    parser.add_argument("--model_name", type=str, help="model name", default="FcNet4")

    parser.add_argument(
        "--training_epoch", type=int, help="total training epoch", default=100
    )
    parser.add_argument(
        "--batch_size", type=int, help="input size for training", default=50
    )

    # ----------------------------------------------------------------------------------------------------
    # META param
    # ----------------------------------------------------------------------------------------------------
    parser.add_argument(
        "--max_inner", type=int, help="number of inner loop", default=15
    )
    parser.add_argument("--max_outer", type=int, help="number of outer loop", default=5)
    parser.add_argument(
        "--lr_prior",
        type=float,
        help="learning rate for prior model (0.5-1)",
        default=0.01,  # 0.01
    )
    parser.add_argument(
        "--lr_post", type=float, help="learning rate for post model", default=0.01  # 0.4
    )
    # parser.add_argument(
    #     "--weight",
    #     type=float,
    #     help="weights for controlling ERM and KL divergence (at least 0.1)",
    #     default=0.4,
    # )

    parser.add_argument(
        "--divergence_type",
        type=str,
        help="choose the divergence type 'KL' or 'W_Sqr'",
        default="W_Sqr",
    )
    parser.add_argument(
        "--kappa_prior",
        type=float,
        help="The STD of the 'noise' added to prior while using KL",
        default=0.01,
    )
    parser.add_argument(
        "--kappa_post",
        type=float,
        help="The STD of the 'noise' added to post while using KL",
        default=1e-3,
    )
    # ----------------------------------------------------------------------------------------------------
    # Stochastic
    # ----------------------------------------------------------------------------------------------------
    parser.add_argument(
        "--log_var_init_mean",
        type=float,
        help="Weights initialization (for Bayesian net) - mean",
        default=-0.1,
    )
    parser.add_argument(
        "--log_var_init_var",
        type=float,
        help="Weights initialization (for Bayesian net) - var",
        default=0.1,
    )
    parser.add_argument(
        "--eps_std",
        type=float,
        help="Bayesian Network Noisy Ratio",
        default=0.1,
    )
    parser.add_argument(
        "--n_MC", type=int, help="Number of Monte-Carlo iterations", default=5
    )
    # ----------------------------------------------------------------------------------------------------
    # Post Processing
    # ----------------------------------------------------------------------------------------------------
    parser.add_argument(
        "--acc_bin",
        type=float,
        help="accuracy bar while evaluating predict result",
        default=0.5,
    )
    parser.add_argument(
        "--params",
        type=dict,
        help="param dict for suf calibration gap calaculation",
        default=None,
    )
    # ----------------------------------------------------------------------------------------------------
    # Other
    # ----------------------------------------------------------------------------------------------------
    parser.add_argument("--seed", type=int, help="seed", default=0)
    parser.add_argument(
        "--use_wandb", type=bool, help="whether use_wandb", default=False
    )
    parser.add_argument(
        "--wandb_username", type=str, help="wandb user name", default='UNKNOWN'
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="outpur dir prefix for log info and result",
        default="test",
    )
    parser.add_argument(
        "--train_inf_step", type=int, help="Inference Period while training", default=1
    )

    parser.add_argument(
        "--is_bilevel", type=bool,
        help="whether use bilevel to tackle the imbalance issue",
        default=False
    )

    parser.add_argument(
        "--manual_adjust", type=bool,
        help="whether manually adjust label",
        default=False
    )

    parser.add_argument(
        "--sharp_strategy", type=bool,
        help="whether use sharp strategy to flatten the objective landscape",
        default=True
    )

    parser.add_argument(
        "--rho", type=float,
        help="hyper parameter for sharp strategy",
        default=0.3
    )

    parser.add_argument(
        "--method", type=int,
        help="1: FAMS, 2: FAMS+manual logits adjustment"
             "3: Ours, 4: Ours-KL in up level"
             "5: Ours-indirect grad for global f"
             "6: Ours-KL in up level-indirect grad for global f"
             "7: trivial ERM, 8: Ours with sharp strategy"
             "9: balanced ERM",
        default=9
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------------------------------------------
    # config file update
    # ----------------------------------------------------------------------------------------------------

    if args.config:
        cfg_dir = osp.abspath(osp.join(osp.dirname(__file__), args.config))
        opt = vars(args)
        args = yaml.load(open(cfg_dir), Loader=yaml.FullLoader)
        opt.update(args)
        args = argparse.Namespace(**opt)

    main(args)
