from __future__ import absolute_import, division, print_function

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
from sklearn.metrics import balanced_accuracy_score

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

        optimizer_post.zero_grad()  # position changed by Bojian, once before loss.backward()

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

        loss = avg_empiric_loss + prm.weight * complexity

        loss.backward()
        optimizer_post.step()

        # freeze the posterior model
        model_freeze(post_model)
        # log
        # if _ + 1 == prm.max_inner:
        if (_ + 1) % 2 == 0:
            model_num = prm.model_num if prm.method == "ours_sample" else prm.N_subtask
            logger.info(
                "Epoch=[{}/{}], Inner Loop Task=[{}/{}], Inner Step=[{}/{}], CLS_LOSS = {}, DIS_LOSS = {}, TOTAL_LOSS = {}".format(
                    epoch_id + 1,
                    prm.training_epoch,
                    task_index + 1,
                    model_num,
                    _ + 1,
                    prm.max_inner,
                    avg_empiric_loss.item(),
                    complexity.item(),
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
    for _ in range(prm.max_inner):

        # model_freeze(prior_model)
        model_activate(prior_model)
        model_activate(post_model)
        post_model.train()

        avg_empiric_loss = 0

        # number of MC sample (default as 5)
        n_MC = prm.n_MC
        inputs, targets = batch

        optimizer_post.zero_grad()  # position changed by Bojian, once before loss.backward()

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

        loss = avg_empiric_loss + prm.weight * complexity
        # added by Bojian
        d_L_low_d_post += gather_flat_grad(grad(loss, post_model.parameters(), create_graph=True))
        # second_derivative = gather_flat_grad(grad(d_L_low_d_post, post_model.parameters(),
        #      grad_outputs=torch.ones(num_weights, device=prm.device), retain_graph=True))

        # loss.backward()
        # optimizer_post.step()

        # freeze the posterior model
        # model_freeze(post_model)
        # log
        # if _ + 1 == prm.max_inner:
        if (_ + 1) % 2 == 0:
            model_num = prm.model_num if prm.method == "ours_sample" else prm.N_subtask
            logger.info(
                "Epoch=[{}/{}], Inner Loop Task=[{}/{}], Inner Step=[{}/{}], CLS_LOSS = {}, DIS_LOSS = {}, TOTAL_LOSS = {}".format(
                    epoch_id + 1,
                    prm.training_epoch,
                    task_index + 1,
                    model_num,
                    _ + 1,
                    prm.max_inner,
                    avg_empiric_loss.item(),
                    complexity.item(),
                    loss.item(),
                )
            )
            if prm.use_wandb and task_index <= 5:
                name = prm.method + "_task_" + str(task_index) + "_all_loss"
                wandb.log({name: loss.item()}, commit=False)

    return d_L_low_d_post / prm.max_inner  # added by Bojian


def update_meta_prior(list_of_post_model,
                      list_optimizer_post,  # added by Bojian
                      prior_model,
                      optimizer_hyper,  # added by Bojian
                      up_loss_criterion,  # added by Bojian
                      list_d_L_low_d_post,  # added by Bojian
                      val_loader,  # added by Bojian
                      up_params,  # added by Bojian
                      optimizer_prior, prm, epoch_id):
    for _ in range(prm.max_outer):

        model_activate(prior_model)
        for post_model in list_of_post_model:
            model_activate(post_model)
            post_model.train()

        complexity = 0.0
        kld_list = []
        complexity_list = []
        val_loss_list = []
        up_loss_list = []

        # back+prob with posterior

        # prior_model.train()
        for optimizer_post in list_optimizer_post:
            optimizer_post.zero_grad()
        optimizer_prior.zero_grad()
        optimizer_hyper.zero_grad()

        num_weights_prior = sum(p.numel() for p in prior_model.parameters())
        num_weights_post = sum(p.numel() for p in list_of_post_model[0].parameters())
        d_L_up_d_prior = torch.zeros(num_weights_prior, device=prm.device)
        list_d_L_up_d_post = [torch.zeros(num_weights_post, device=prm.device)
                              for _ in list_of_post_model]

        for val_data, val_targets in val_loader:
            val_data, val_targets = val_data.to(prm.device), val_targets.to(prm.device)
            val_output = prior_model(val_data)
            val_loss = up_loss_criterion(val_output, val_targets, up_params)
            val_loss_list.append(val_loss.detach().numpy())
            # calculate KL divergence of prior model with all post models
            complexity = 0
            for idx_post, post_model in enumerate(list_of_post_model):
                kld = get_KLD(prior_model, post_model, prm, noised_prior=True)
                complexity = complexity + kld
                kld_list.append(str(round(kld.item(), 4)))
                list_optimizer_post[idx_post].zero_grad()
                list_d_L_up_d_post[idx_post] += \
                    gather_flat_grad(grad(kld, post_model.parameters(), retain_graph=True))

            complexity = complexity / len(list_of_post_model)
            complexity_list.append(complexity.detach().numpy())

            up_loss = val_loss + prm.weight * complexity
            up_loss_list.append(up_loss.detach().numpy())

            optimizer_prior.zero_grad()
            d_L_up_d_prior += \
                gather_flat_grad(grad(up_loss, prior_model.parameters(), retain_graph=True))
        d_L_up_d_prior /= len(val_loader)
        list_d_L_up_d_post = [d_L_up_d_post / len(val_loader)
                              for d_L_up_d_post in list_d_L_up_d_post]

        preconditioner = neumann_hyperstep_preconditioner(
            torch.mean(torch.stack(list_d_L_up_d_post), dim=0),
            list_d_L_low_d_post, 1.0, 5,
            list_of_post_model, prm)

        indirect_grad_prior = torch.zeros(num_weights_prior, device=prm.device)
        for idx, d_L_low_d_post in enumerate(list_d_L_low_d_post):
            prior_model.zero_grad()
            indirect_grad_prior += gather_flat_grad(
                grad(d_L_low_d_post,
                     prior_model.parameters(),
                     grad_outputs=preconditioner.view(-1),
                     retain_graph=True,
                     create_graph=True,
                     allow_unused=True))

        prior_grad = d_L_up_d_prior - indirect_grad_prior

        indirect_grad_hyper = torch.zeros(4, device=prm.device)
        for d_L_low_d_post in list_d_L_low_d_post:
            optimizer_hyper.zero_grad()
            indirect_grad_hyper += gather_flat_grad(
                grad(d_L_low_d_post,
                     get_trainable_hyper_params(up_params),
                     grad_outputs=preconditioner.view(-1),
                     retain_graph=True,
                     create_graph=True,
                     allow_unused=True))

        hyper_grad = -indirect_grad_hyper

        # complexity.backward()
        optimizer_hyper.zero_grad()
        assign_gradient(up_params, hyper_grad, prm.num_classes)
        optimizer_hyper.step()

        optimizer_prior.zero_grad()
        assign_gradient(prior_model.parameters(), prior_grad, prm.num_classes)
        optimizer_prior.step()

        # if _ + 1 == prm.max_outer:
        if (_ + 1) % 1 == 0:
            # complexity_str = "; ".join(complexity_list)

            logger.info(
                "Epoch=[{}/{}], Outer Loop Step=[{}/{}], "
                "Mean Complexity = {}, Mean Val Loss = {}, Mean Up Loss = {}".format(
                    epoch_id + 1, prm.training_epoch, _ + 1, prm.max_outer,
                    np.mean(complexity_list), np.mean(val_loss_list), np.mean(up_loss_list)
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


def update_meta_post(list_of_post_model, prior_model, ratio):
    for i in range(len(list_of_post_model)):
        prob = random.uniform(0, 1)
        if prob < ratio:
            list_of_post_model[i] = copy.deepcopy(prior_model)


def train_ours(prm, prior_model,
               low_loss_criterion,  # added by Bojian
               up_loss_criterion,  # added by Bojian
               # loss_criterion,     # commented by Bojian
               X_train, A_train, y_train,
               X_val, A_val, y_val,  # added by Bojian
               X_test=None, A_test=None, y_test=None):
    # prior model
    optimizer_prior = optim.Adagrad(prior_model.parameters(), lr=prm.lr_prior)
    # optimizer schedular
    optimizer_prior_schedular = PriorExponentialLR(
        optimizer_prior, prm.training_epoch)

    # optimizer for hyperparameter, added by Bojian
    # num_classes = len(np.unique(y_train))
    dy = get_init_dy(prm.device, num_classes=prm.num_classes)  # multiplicative adjustments to the logit
    ly = get_init_ly(prm.device, num_classes=prm.num_classes)  # additive adjustments to the logit
    w_train = get_train_w(prm.device, num_classes=prm.num_classes)  # weight for cross entropy for different class
    class_num_val = []
    for i in range(prm.num_classes):
        class_num_val.append(np.sum(y_test == i))
    w_val = get_val_w(prm.device, class_num_val)  # weight for cross entropy for different class

    low_params = [dy, ly, w_train]
    up_params = [dy, ly, w_val]

    optimizer_hyper = optim.SGD(params=[{'params': dy}, {'params': ly}],
                                lr=0.01, momentum=0.9, weight_decay=5e-4)
    # optimizer schedular
    optimizer_hyper_schedular = PriorExponentialLR(
        optimizer_hyper, prm.training_epoch)

    # post model
    model_num = prm.N_subtask
    post_models = [get_model(prm) for _ in range(model_num)]
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
            sample_batch_sen_idx(
                X_train, A_train, y_train, prm, np.unique(A_train)[
                    t_num]
            )
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

        time_e = time.time()

        if epoch_id % prm.train_inf_step == 0:
            ss_time = time_e - time_s
            logger.info(
                "The training time for one epoch is: {} seconds".format(
                    ss_time)
            )
            predict = inference(prior_model, X_test, prm)
            accuracy, b_acc = compute_accuracy(y_test, predict, 0.5, prm.output_dim)
            # b_acc = balanced_accuracy_score(y_test, predict.argmax(1))
            logger.info(
                "The overall accuracy of EPOCH [{}] is: {}".format(
                    epoch_id, accuracy)
            )
            logger.info(
                "The overall balanced accuracy of EPOCH [{}] is: {}".format(
                    epoch_id, b_acc)
            )
            if prm.use_wandb:
                wandb_dict = result_wandb(y_test, predict, A_test, prm)
                wandb.log(wandb_dict, commit=False)

            if epoch_id == int(prm.training_epoch / 2):
                npy_dir, npy_file_pre = set_npy(prm, epoch_id)

                if not os.path.exists(npy_dir):
                    os.makedirs(npy_dir)

                np.save(osp.join(npy_dir, npy_file_pre + "_testy.npy"), y_test)
                np.save(osp.join(npy_dir, npy_file_pre + "_testA.npy"), A_test)
                np.save(osp.join(npy_dir, npy_file_pre + "_predict.npy"), predict)


def train(
        prm,
        prior_model,
        low_loss_criterion,  # added by Bojian
        up_loss_criterion,  # added by Bojian
        # loss_criterion,    # commented by Bojian
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

    # initial test
    predict = inference(prior_model, X_test, prm)
    accuracy, b_acc = compute_accuracy(y_test, predict, 0.5, prm.output_dim)
    # b_acc = balanced_accuracy_score(y_test, predict.argmax(1))
    logger.info("The Initial overall accuracy is: {}".format(accuracy))
    logger.info("The Initial overall balanced accuracy is: {}".format(b_acc))
    if prm.use_wandb:
        wandb_dict = result_wandb(y_test, predict, A_test, prm)
        wandb.log(wandb_dict)

    # wandb.finish()

    # training switch

    # train_ours(prm, prior_model, loss_criterion, X_train,
    #             A_train, y_train, X_test, A_test, y_test)
    train_ours(prm, prior_model, low_loss_criterion, up_loss_criterion,
               X_train, A_train, y_train,
               X_val, A_val, y_val,  # added by Bojian
               X_test, A_test, y_test)

    if prm.use_wandb:
        wandb.finish()


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
