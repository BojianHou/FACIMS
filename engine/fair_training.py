from __future__ import absolute_import, division, print_function

import datetime
import logging
import torch
from torch.autograd import grad
from torch.utils.data import DataLoader
from utils.complexity_terms import get_KLD
import numpy as np
from layers.stochastic_models import get_model
import torch.optim as optim
from data.dataset import sample_batch_sen_idx
from utils.postprocessing import compute_accuracy, result_wandb
from utils.optim import PostMultiStepLR, PriorExponentialLR
from utils.loggers import set_wandb, set_npy
import wandb
import copy
import os
import os.path as osp
import random
import time
from utils.common import get_init_dy, get_init_ly, get_train_w, get_val_w
from utils.common import gather_flat_grad, neumann_hyperstep_preconditioner
from utils.common import get_trainable_hyper_params, assign_gradient
from utils.common import model_activate, model_freeze
from utils.postprocessing import demographic_parity_binary, equalized_odds_binary, standard_suf_gap_all_binary
from utils.postprocessing import demographic_parity_multiclass, equalized_odds_multiclass, standard_suf_gap_all_multiclass
from utils.common import cross_entropy, loss_adjust_cross_entropy
from utils.common import loss_adjust_cross_entropy_manual
from utils.sharp_strategy import SAM
import torch.nn.functional as F

logger = logging.getLogger("fair")


# ----------------------------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------------------------


# main training
def train_one_task(
        prior_model,
        post_model,
        loss_criterion,
        optimizer_post,
        batch,
        prm,
        task_index,
        epoch_id,
        low_params  # added by Bojian
):
    """
    :param prior_model: the prior_model (only one-model)
    :param post_model: the posterior_model within task a
    :param loss_criterion: the loss function
    :param optimizer_post: optimizer for the post model (contains a list of optimizers)
    :param batch: the data batch that contains t-tasks.
    :param prm: parameter configuration
    :return:
    """

    # num_weights = sum(p.numel() for p in post_model.parameters())
    # d_L_low_d_post = torch.zeros(num_weights, device=prm.device)  # gradient of post model w.r.t. L_low
    logger.info('--------------Optimize the post models with gradient descent------------')
    for _ in range(prm.max_inner):

        model_freeze(prior_model)
        model_activate(post_model)
        post_model.train()

        avg_empiric_loss = 0

        # number of MC sample (default as 5)
        n_MC = prm.n_MC
        inputs, targets = batch

        # Monte-Carlo loop in estimation prediction loss
        for i_MC in range(n_MC):
            # Empirical Loss on current task:
            outputs = post_model(inputs)
            avg_empiric_loss_curr = loss_criterion(outputs, targets, low_params)
            avg_empiric_loss = avg_empiric_loss + \
                               (1 / n_MC) * avg_empiric_loss_curr
        # end Monte-Carlo loop

        # Compute the complexity term (noised prior can be set false)
        complexity = get_KLD(prior_model, post_model, prm, noised_prior=True)

        # loss = prm.lambda_low * avg_empiric_loss + (1 - prm.lambda_low) * complexity
        loss = avg_empiric_loss + prm.lambda_low * complexity

        optimizer_post.zero_grad()
        loss.backward()  # calculate the gradient for the original parameters w
        if prm.sharp_strategy:
            optimizer_post.first_step(zero_grad=True)  # calculate the epsilon and add it to w
            # use the disturbed new w to do forward process
            avg_empiric_loss = 0
            for i_MC in range(n_MC):
                # Empirical Loss on current task:
                outputs = post_model(inputs)
                avg_empiric_loss_curr = loss_criterion(outputs, targets, low_params)
                avg_empiric_loss = avg_empiric_loss + \
                                   (1 / n_MC) * avg_empiric_loss_curr

            # Compute the complexity term (noised prior can be set false)
            complexity = get_KLD(prior_model, post_model, prm, noised_prior=True)

            # loss = prm.lambda_low * avg_empiric_loss + (1 - prm.lambda_low) * complexity
            loss = avg_empiric_loss + prm.lambda_low * complexity
            optimizer_post.zero_grad()
            loss.backward()
            optimizer_post.second_step(zero_grad=True)
        else:
            optimizer_post.step()

        # freeze the posterior model
        model_freeze(post_model)
        # log
        if _ + 1 == prm.max_inner:
        # if (_ + 1) % 2 == 0:
            model_num = prm.model_num if prm.method == "ours_sample" else prm.N_subtask
            logger.info(
                "Epoch=[{}/{}], Inner Loop Task=[{}/{}], Inner Step=[{}/{}], "
                "DIS_LOSS = {:.4f}, CLS_LOSS = {:.4f}, TOTAL_LOSS = {:.4f}".format(
                    epoch_id + 1,
                    prm.training_epoch,
                    task_index + 1,
                    model_num,
                    _ + 1,
                    prm.max_inner,
                    complexity.item(),
                    avg_empiric_loss.item(),
                    loss.item(),
                )
            )
            if prm.use_wandb and task_index <= 5:
                name = prm.method + "_task_" + str(task_index) + "_all_loss"
                wandb.log({name: loss.item()}, commit=False)

    # return d_L_low_d_post/prm.max_inner, post_model  # added by Bojian


def train_one_task_without_GD(
        prior_model,
        post_model,
        loss_criterion,
        optimizer_post,
        batch,
        prm,
        task_index,
        epoch_id,
        low_params  # added by Bojian
):
    """
    :param prior_model: the prior_model (only one-model)
    :param post_model: the posterior_model within task a
    :param loss_criterion: the loss function
    :param optimizer_post: optimizer for the post model (contains a list of optimizers)
    :param batch: the data batch that contains t-tasks.
    :param prm: parameter configuration
    :return:
    """

    num_weights = sum(p.numel() for p in post_model.parameters())
    d_L_low_d_post = torch.zeros(num_weights, device=prm.device)  # gradient of post model w.r.t. L_low

    logger.info('--------------Get the loss of the post models without doing gradient descent------------')
    max_inner = 1
    for _ in range(max_inner):

        # model_freeze(prior_model)
        model_activate(prior_model)
        model_activate(post_model)
        post_model.train()

        avg_empiric_loss = 0

        # number of MC sample (default as 5)
        n_MC = prm.n_MC
        inputs, targets = batch

        # Monte-Carlo loop in estimation prediction loss
        for i_MC in range(n_MC):
            # Empirical Loss on current task:
            outputs = post_model(inputs)
            avg_empiric_loss_curr = loss_criterion(outputs, targets, low_params)
            avg_empiric_loss = avg_empiric_loss + \
                               (1 / n_MC) * avg_empiric_loss_curr
        # end Monte-Carlo loop

        # Compute the complexity term (noised prior can be set false)
        complexity = get_KLD(prior_model, post_model, prm, noised_prior=True)

        # loss = prm.lambda_low * avg_empiric_loss + (1 - prm.lambda_low) * complexity
        loss = avg_empiric_loss + prm.lambda_low * complexity
        # added by Bojian
        optimizer_post.zero_grad()
        d_L_low_d_post += gather_flat_grad(grad(loss, post_model.parameters(), create_graph=True))
        # second_derivative = gather_flat_grad(grad(d_L_low_d_post, post_model.parameters(),
        #      grad_outputs=torch.ones(num_weights, device=prm.device), retain_graph=True))

        # loss.backward()
        # optimizer_post.step()

        # freeze the posterior model
        # model_freeze(post_model)
        # log
        # if _ + 1 == prm.max_inner:
        # if (_ + 1) % 2 == 0:
        model_num = prm.model_num if prm.method == "ours_sample" else prm.N_subtask
        logger.info(
            "Epoch=[{}/{}], Inner Loop Task=[{}/{}], Inner Step=[{}/{}], "
            "DIS_LOSS = {:.4f}, CLS_LOSS = {:.4f}, TOTAL_LOSS = {:.4f}".format(
                epoch_id + 1,
                prm.training_epoch,
                task_index + 1,
                model_num,
                _ + 1,
                max_inner,
                complexity.item(),
                avg_empiric_loss.item(),
                loss.item(),
            )
        )
        if prm.use_wandb and task_index <= 5:
            name = prm.method + "_task_" + str(task_index) + "_all_loss"
            wandb.log({name: loss.item()}, commit=False)

    return d_L_low_d_post / max_inner  # added by Bojian


def update_meta_prior(list_of_post_model,
                      list_optimizer_post,  # added by Bojian
                      prior_model,
                      optimizer_hyper,  # added by Bojian
                      up_loss_criterion,  # added by Bojian
                      list_d_L_low_d_post,  # added by Bojian
                      val_loader,  # added by Bojian
                      up_params,  # added by Bojian
                      optimizer_prior, prm, epoch_id):

    logger.info('--------------Optimize the prior model-------------')
    for _ in range(prm.max_outer):

        model_activate(prior_model)
        for post_model in list_of_post_model:
            model_activate(post_model)
            post_model.train()

        kld_list = []

        # prior_model.train()
        # for optimizer_post in list_optimizer_post:
        #     optimizer_post.zero_grad()
        #
        # optimizer_prior.zero_grad()
        # optimizer_hyper.zero_grad()

        if prm.is_bilevel:
            # initialize the gradient of L_up w.r.t prior model and post models
            num_weights_prior = sum(p.numel() for p in prior_model.parameters())
            num_weights_post = sum(p.numel() for p in list_of_post_model[0].parameters())
            d_L_up_d_prior = torch.zeros(num_weights_prior, device=prm.device)
            list_d_L_up_d_post = [torch.zeros(num_weights_post, device=prm.device)
                                  for _ in list_of_post_model]

        # calculate KL divergence of prior model with all post models
        complexity = 0
        for idx_post, post_model in enumerate(list_of_post_model):
            kld = get_KLD(prior_model, post_model, prm, noised_prior=True)
            complexity = complexity + kld
            kld_list.append(str(round(kld.item(), 4)))
            if prm.is_bilevel:
                list_optimizer_post[idx_post].zero_grad()
                list_d_L_up_d_post[idx_post] += \
                    gather_flat_grad(grad(kld, post_model.parameters(), retain_graph=True))
        complexity = complexity / len(list_of_post_model)

        if not prm.is_bilevel:  # FAMS original code, directly use KL divergence as loss to bp prior model
            optimizer_prior.zero_grad()
            complexity.backward()
            optimizer_prior.step()
        else:  # calculate the direct and indirect gradient for prior model and hyperparameters
            val_loss = 0
            for val_data, val_targets in val_loader:
                val_data, val_targets = val_data.to(prm.device), val_targets.to(prm.device)
                val_output = prior_model(val_data)
                val_loss += up_loss_criterion(val_output, val_targets, up_params)
            val_loss = val_loss / len(val_loader)
            if prm.no_KL:
                up_loss = val_loss
            elif prm.no_val:
                up_loss = complexity
            else:
                up_loss = prm.lambda_up * val_loss + (1-prm.lambda_up) * complexity

            optimizer_prior.zero_grad()
            d_L_up_d_prior += \
                gather_flat_grad(grad(up_loss, prior_model.parameters(), retain_graph=True))

            # d_L_up_d_prior /= len(val_loader)
            # list_d_L_up_d_post = [d_L_up_d_post / len(val_loader)
            #                       for d_L_up_d_post in list_d_L_up_d_post]

            # *important*
            # preconditioner is the multiplication of the mean of d_L_up_d_post
            # and the inverse of the square of the d_L_low_d_post
            preconditioner = neumann_hyperstep_preconditioner(
                torch.mean(torch.stack(list_d_L_up_d_post), dim=0),
                list_d_L_low_d_post, 0.1, 10,
                list_of_post_model, prm)

            if prm.no_indirect_grad:
                prior_grad = d_L_up_d_prior  # direct gradient
            else:
                indirect_grad_prior = torch.zeros(num_weights_prior, device=prm.device)
                for d_L_low_d_post in list_d_L_low_d_post:
                    prior_model.zero_grad()
                    indirect_grad_prior += gather_flat_grad(
                        grad(d_L_low_d_post,
                             prior_model.parameters(),
                             grad_outputs=preconditioner.view(-1),
                             # retain_graph=True,
                             create_graph=True,
                             allow_unused=True))

                prior_grad = prm.lambda_up * d_L_up_d_prior - (1 - prm.lambda_up) * indirect_grad_prior

            if not prm.no_val:  # no val means no need to tune the hyperparameters
                # we have two hyperparameters, ly and dy,
                # each of which has the dimension of output_dim
                indirect_grad_hyper = torch.zeros(2 * prm.output_dim, device=prm.device)
                for d_L_low_d_post in list_d_L_low_d_post:
                    optimizer_hyper.zero_grad()
                    indirect_grad_hyper += gather_flat_grad(
                        grad(d_L_low_d_post,
                             get_trainable_hyper_params(up_params),
                             grad_outputs=preconditioner.view(-1),
                             # retain_graph=True,
                             create_graph=True,
                             allow_unused=True))

                hyper_grad = -indirect_grad_hyper

                optimizer_hyper.zero_grad()
                assign_gradient(up_params, hyper_grad, prm.num_classes)
                optimizer_hyper.step()

            # complexity.backward()
            optimizer_prior.zero_grad()
            assign_gradient(prior_model.parameters(), prior_grad, prm.num_classes)
            optimizer_prior.step()


        if _ + 1 == prm.max_outer:
        # if (_ + 1) % 1 == 0:
            # complexity_str = "; ".join(complexity_list)

            if prm.is_bilevel:
                logger.info(
                    "Epoch=[{}/{}], Outer Loop Step=[{}/{}], "
                    "Complexity = {:.4f}, Val Loss = {:.4f}, All Up Loss = {:.4f}".format(
                        epoch_id + 1, prm.training_epoch, _ + 1, prm.max_outer,
                        complexity, val_loss, up_loss
                    )
                )
            else:
                logger.info(
                    "Epoch=[{}/{}], Outer Loop Step=[{}/{}], "
                    "Complexity = {:.4f}".format(
                        epoch_id + 1, prm.training_epoch, _ + 1, prm.max_outer,
                        complexity
                    )
                )

            if prm.use_wandb:
                wandb.log(
                    {"prior_model_loss": complexity.item() / prm.N_subtask})


def training_task_batches(
        list_of_post_model,
        list_of_post_optimizer,
        prior_model,
        prior_optimizer,
        optimizer_hyper,  # added by Bojian
        low_loss_criterion,  # added by Bojian
        up_loss_criterion,  # added by Bojian
        # loss_criterion,    # commented by Bojian
        train_batch,
        val_loader,  # added by Bojian
        prm,
        epoch_id,
        optimizer_prior_schedular,
        list_optimizer_post_schedular,
        optimizer_hyper_schedular,  # added by Bojian
        low_params, up_params  # added by Bojian
):
    """
    :param list_of_post_model:
    :param list_of_post_optimizer:
    :param prior_model:
    :param prior_optimizer:
    :param train_batch:
    :param low_loss_criterion:
    :param prm:
    :return:
    """

    list_d_L_low_d_post = []
    # new_list_post_model = []
    for task_index, data in enumerate(train_batch):
        post_model = list_of_post_model[task_index]
        posterior_opt = list_of_post_optimizer[task_index]

        # training task
        # d_L_low_d_post = train_one_task(
        train_one_task(
            prior_model,
            post_model,
            low_loss_criterion,  # added by Bojian
            posterior_opt,
            data,
            prm,
            task_index,
            epoch_id,
            low_params  # added by Bojian
        )

        # new_list_post_model.append(post_model)
        list_optimizer_post_schedular[task_index].step()

        if prm.is_bilevel:
            d_L_low_d_post = train_one_task_without_GD(
                prior_model,
                post_model,
                low_loss_criterion,  # added by Bojian
                posterior_opt,
                data,
                prm,
                task_index,
                epoch_id,
                low_params  # added by Bojian
            )
            list_d_L_low_d_post.append(d_L_low_d_post)


    # Then updating prior distribution
    update_meta_prior(list_of_post_model,
                      list_of_post_optimizer,
                      prior_model,
                      optimizer_hyper,  # added by Bojian
                      up_loss_criterion,  # added by Bojian
                      list_d_L_low_d_post,  # added by Bojian
                      val_loader,  # added by Bojian
                      up_params,  # added by Bojian
                      prior_optimizer, prm, epoch_id)

    if prm.dataset in ["adult"]:
        if epoch_id >= 60:
            optimizer_prior_schedular.step()
    elif prm.dataset in ["celeba"]:
        pass
    else:
        optimizer_prior_schedular.step()

    logger.info(
        f"current prior optimizer lr={prior_optimizer.param_groups[0]['lr']}")
    logger.info(
        f"current post optimizer lr={list_of_post_optimizer[0].param_groups[0]['lr']}"
    )
    logger.info(
        f"current hyper optimizer lr={optimizer_hyper.param_groups[0]['lr']}"
    )


def update_meta_post(list_of_post_model, prior_model, ratio):
    for i in range(len(list_of_post_model)):
        prob = random.uniform(0, 1)
        if prob < ratio:
            list_of_post_model[i] = copy.deepcopy(prior_model)


def train_ours(prm,
               X_train, A_train, y_train,
               X_val, A_val, y_val,  # added by Bojian
               X_test=None, A_test=None, y_test=None):

    # Training Components
    # loss_criterion = nn.BCELoss()
    if prm.manual_adjust:
        low_loss_criterion = loss_adjust_cross_entropy_manual
    else:
        low_loss_criterion = loss_adjust_cross_entropy
    up_loss_criterion = cross_entropy

    # create the prior model
    prior_model = get_model(prm)

    # initial test
    predict = inference(prior_model, X_test, prm)
    accuracy, b_acc, recall = compute_accuracy(y_test, predict, 0.5, prm.output_dim)
    logger.info("The initial overall accuracy is:          {:.4f}".format(accuracy))
    logger.info("The initial overall balanced accuracy is: {:.4f}".format(b_acc))
    if prm.use_wandb:
        wandb_dict = result_wandb(y_test, predict, A_test, prm.output_dim)
        wandb.log(wandb_dict)

    # optimizer for prior model
    # if prm.sharp_strategy:
    #     optimizer_prior = SAM(base_optimizer=optim.Adagrad, params=prior_model.parameters(), lr=prm.lr_prior)
    # else:
    optimizer_prior = optim.Adagrad(prior_model.parameters(), lr=prm.lr_prior)

    # optimizer schedular
    optimizer_prior_schedular = PriorExponentialLR(
        optimizer_prior, prm.training_epoch)

    # optimizer for hyperparameter, added by Bojian
    # num_classes = len(np.unique(y_train))
    dy = get_init_dy(prm.device, num_classes=prm.output_dim)  # multiplicative adjustments to the logit
    ly = get_init_ly(prm.device, num_classes=prm.output_dim)  # additive adjustments to the logit
    # w_train = get_train_w(prm.device, num_classes=prm.num_classes)  # weight for cross entropy for different class
    class_num_val = []
    for i in range(prm.num_classes):
        class_num_val.append(np.sum(y_test == i))
    w_val = get_val_w(prm.device, class_num_val)  # weight for cross entropy for different class
    w_train = w_val.clone()
    if prm.manual_adjust:
        low_params = w_val
    else:
        low_params = [dy, ly, w_train]
    up_params = [dy, ly, w_val]

    # if prm.sharp_strategy:
    #     optimizer_hyper = SAM(base_optimizer=optim.SGD, params=[{'params': dy}, {'params': ly}],
    #                                 lr=0.001, momentum=0.9, weight_decay=5e-4)
    # else:
    optimizer_hyper = optim.SGD(params=[{'params': dy}, {'params': ly}],
                                lr=0.001, momentum=0.9, weight_decay=5e-4)
    # optimizer schedular
    optimizer_hyper_schedular = PriorExponentialLR(
        optimizer_hyper, prm.training_epoch)

    # post model
    model_num = prm.N_subtask
    post_models = [get_model(prm) for _ in range(model_num)]

    if prm.sharp_strategy:
        rho_list = [0.00005 * (10**item) for item in range(prm.N_subtask)]  # small rho for majority group
        list_optimizer_post = [
            SAM(base_optimizer=optim.Adagrad, rho=rho, params=post_model.parameters(), lr=prm.lr_post)
            for rho, post_model in zip(rho_list, post_models)
        ]
    else:
        list_optimizer_post = [
            optim.Adagrad(post_model.parameters(), lr=prm.lr_post)
            for post_model in post_models
        ]
    list_optimizer_post_schedular = [
        PostMultiStepLR(list_optimizer_post[i], prm.training_epoch)
        for i in range(model_num)
    ]

    for epoch_id in range(prm.training_epoch):

        batch_train = [
            sample_batch_sen_idx(X_train, A_train, y_train, prm, np.unique(A_train)[t_num])
            for t_num in range(prm.N_subtask)
        ]

        data_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(torch.from_numpy(X_val).float()),
            torch.LongTensor(y_val))
        val_loader = DataLoader(data_val, batch_size=50, shuffle=True)

        time_s = time.time()

        training_task_batches(
            post_models,
            list_optimizer_post,
            prior_model,
            optimizer_prior,
            optimizer_hyper,  # added by Bojian
            low_loss_criterion,  # added by Bojian
            up_loss_criterion,  # added by Bojian
            # loss_criterion,    # commented by Bojian
            batch_train,  # altered by Bojian
            val_loader,  # added by Bojian
            prm,
            epoch_id,
            optimizer_prior_schedular,
            list_optimizer_post_schedular,
            optimizer_hyper_schedular,
            low_params, up_params
        )
        torch.cuda.empty_cache()

        time_e = time.time()

        if epoch_id % prm.train_inf_step == 0:
            ss_time = time_e - time_s
            logger.info(
                "The training time for one epoch is: {}".format(
                    str(datetime.timedelta(seconds=ss_time)))
            )

            predict = inference(prior_model, X_test, prm)

            result_show_in_epoch(y_test, predict, A_test, prm.output_dim, epoch_id)

            if prm.use_wandb:
                wandb_dict = result_wandb(y_test, predict, A_test, prm.output_dim)
                wandb.log(wandb_dict, commit=False)

            # if epoch_id == int(prm.training_epoch / 2):
            #     npy_dir, npy_file_pre = set_npy(prm, epoch_id)
            #
            #     if not os.path.exists(npy_dir):
            #         os.makedirs(npy_dir)
            #
            #     np.save(osp.join(npy_dir, npy_file_pre + "_testy.npy"), y_test)
            #     np.save(osp.join(npy_dir, npy_file_pre + "_testA.npy"), A_test)
            #     np.save(osp.join(npy_dir, npy_file_pre + "_predict.npy"), predict)

    return (prior_model, post_models)


def train_ERM(prm,
              X_train, A_train, y_train,
              X_val, A_val, y_val,  # added by Bojian
              X_test, A_test, y_test):

    logger.info('==================train trivial ERM===================')
    model_erm = get_model(prm, model_type='Standard')
    optimizer_erm = optim.Adagrad(model_erm.parameters(), lr=0.001)
    data_train = torch.utils.data.TensorDataset(
        torch.FloatTensor(torch.from_numpy(X_train).float()),
        torch.LongTensor(y_train))
    train_loader = DataLoader(data_train, batch_size=50, shuffle=True)

    model_erm.train()
    for epoch_id in range(prm.training_epoch):

        time_s = time.time()
        model_activate(model_erm)
        for data, targets in train_loader:
            data, targets = data.to(prm.device), targets.to(prm.device)
            output = model_erm(data)
            loss = cross_entropy(output, targets)
            loss.backward()
            optimizer_erm.step()

        time_e = time.time()
        if epoch_id % prm.train_inf_step == 0:
            ss_time = time_e - time_s
            logger.info("The training time for one epoch is: {}".format(
                    str(datetime.timedelta(seconds=ss_time))))
            logger.info(
                "Epoch=[{}/{}], Loss={:.4f}".format(
                    epoch_id + 1, prm.training_epoch, loss.item()
                )
            )
            predict = inference(model_erm, X_test, prm)
            result_show_in_epoch(y_test, predict, A_test, prm.output_dim, epoch_id)

    return model_erm


def train_BERM(prm,
              X_train, A_train, y_train,
              X_val, A_val, y_val,  # added by Bojian
              X_test, A_test, y_test):

    logger.info('==================train Balanced ERM===================')
    model_berm = get_model(prm, model_type='Standard')
    optimizer_berm = optim.Adagrad(model_berm.parameters(), lr=0.001)
    data_train = torch.utils.data.TensorDataset(
        torch.FloatTensor(torch.from_numpy(X_train).float()),
        torch.LongTensor(y_train))
    train_loader = DataLoader(data_train, batch_size=50, shuffle=True)

    class_num = []
    for i in range(prm.num_classes):
        class_num.append(np.sum(y_test == i))
    w = get_val_w(prm.device, class_num)  # weight for cross entropy for different class

    model_berm.train()
    for epoch_id in range(prm.training_epoch):

        time_s = time.time()
        model_activate(model_berm)
        for data, targets in train_loader:
            data, targets = data.to(prm.device), targets.to(prm.device)
            output = model_berm(data)
            loss = F.cross_entropy(output, targets, weight=w)
            loss.backward()
            optimizer_berm.step()

        time_e = time.time()
        if epoch_id % prm.train_inf_step == 0:
            ss_time = time_e - time_s
            logger.info("The training time for one epoch is: {}".format(
                    str(datetime.timedelta(seconds=ss_time))))
            logger.info(
                "Epoch=[{}/{}], Loss={:.4f}".format(
                    epoch_id + 1, prm.training_epoch, loss.item()
                )
            )
            predict = inference(model_berm, X_test, prm)
            result_show_in_epoch(y_test, predict, A_test, prm.output_dim, epoch_id)

    return model_berm


def train(prm,
        X_train, A_train, y_train,
        X_val, A_val, y_val,  # added by Bojian
        X_test=None,
        A_test=None,
        y_test=None,
):
    logger.info("===============Training Process=============")

    # wandb setting
    if prm.use_wandb:
        pre_config, project_name, wandb_name = set_wandb(prm)

        wandb.init(
            project=project_name,
            entity=prm.wandb_username,
            config=pre_config,
            reinit=True,
            name=wandb_name,
        )

    if prm.is_ERM:
        model = train_ERM(prm,
                   X_train, A_train, y_train,
                   X_val, A_val, y_val,  # added by Bojian
                   X_test, A_test, y_test)
        # save model
        torch.save(model.state_dict(),
                   './models/data_{}_method_{}_seed_{}_lr_{}'.
                   format(prm.dataset, prm.method, prm.seed, prm.lr))
    elif prm.is_BERM:
        model = train_BERM(prm,
                  X_train, A_train, y_train,
                  X_val, A_val, y_val,  # added by Bojian
                  X_test, A_test, y_test)
        # save model
        torch.save(model.state_dict(),
                   './models/data_{}_method_{}_seed_{}_lr_{}'.
                   format(prm.dataset, prm.method, prm.seed, prm.lr))
    else:
        model = train_ours(prm,
                   X_train, A_train, y_train,
                   X_val, A_val, y_val,  # added by Bojian
                   X_test, A_test, y_test)
        # save model
        torch.save(model[0].state_dict(),
                   './models/data_{}_method_{}_seed_{}_lr_prior_{}_lr_post_{}'.
                   format(prm.dataset, prm.method, prm.seed, prm.lr_prior, prm.lr_post))

    if prm.use_wandb:
        wandb.finish()

    return model


# ----------------------------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------------------------


def inference(model, inputs, prm, n_Mc=5):
    """
    :param model:
    :param inputs:
    :param n_Mc: iteration of monte-carlo (default = 5)
    :return:
    """
    n_Mc = prm.n_MC
    # inputs = torch.tensor(inputs).cuda().float()
    inputs = torch.tensor(inputs).float().to(prm.device)
    model_freeze(model)
    model.eval()
    output_final = 0

    for i_MC in range(n_Mc):
        outputs = model(inputs)
        output_final += (1 / n_Mc) * outputs

    model_activate(model)
    model.train()
    result = output_final.data.cpu().numpy()

    return result


def result_show_in_epoch(y, y_hat, A_test, output_dim, epoch_id):
    accuracy, b_acc, recall = compute_accuracy(y, y_hat, 0.5, output_dim)

    if output_dim <= 2:
        DP_score = demographic_parity_binary(y_hat, A_test)
        EO_score = equalized_odds_binary(y, y_hat, A_test)
        suf_gap_avg_score = standard_suf_gap_all_binary(y, y_hat, A_test)
    else:
        DP_score = demographic_parity_multiclass(y_hat, A_test)
        EO_score = equalized_odds_multiclass(y, y_hat, A_test)
        suf_gap_avg_score = standard_suf_gap_all_multiclass(y, y_hat, A_test)

    for idx, rec in enumerate(recall):
        logger.info("The recall of EPOCH [{}] for class {}:          {:.4f}".format(epoch_id + 1, idx, rec))
    logger.info("The accuracy of EPOCH [{}] is:                 {:.4f}".format(epoch_id + 1, accuracy))
    logger.info("The balanced acc of EPOCH [{}] is:             {:.4f}".format(epoch_id + 1, b_acc))
    logger.info("The demographic parity score of EPOCH [{}] is: {:.4f}".format(epoch_id + 1, DP_score))
    logger.info("The equalized odds score of EPOCH [{}] is:     {:.4f}".format(epoch_id + 1, EO_score))
    logger.info("The group sufficiency gap of EPOCH [{}] is:    {:.4f}".format(epoch_id + 1, suf_gap_avg_score))
