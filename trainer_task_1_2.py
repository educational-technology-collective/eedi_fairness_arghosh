import numpy as np
import torch
import time
import pandas as pd
import random
from utils import open_json, dump_json, compute_auc, compute_accuracy
from dataset_task_1_2 import LSTMDataset, lstm_collate
from model_task_1_2 import LSTMModel, AttentionModel, NCFModel
from copy import deepcopy
from pathlib import Path
import argparse
from shutil import copyfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_acc, best_epoch, best_run = None, -1, None

import mlflow
from mlflow.models import infer_signature


def train_model(model, train_loader, optimizer, epoch):
    mlflow.log_param("epoch",epoch)
    global best_acc, best_epoch, best_run
    batch_idx = 0
    model.train()
    N = len(train_loader.dataset)
    train_loss, all_preds, all_targets = 0., [], []
    val_preds, val_targets = [], []

    for batch in train_loader:
        optimizer.zero_grad()
        loss, output = model(batch)
        #
        if model.task == '1':
            target = batch['labels'].numpy()
            valid_mask = batch['valid_mask'].numpy()
            test_mask = batch['test_mask'].numpy()
            validation_flag = (1-valid_mask)*test_mask
            training_flag = test_mask*valid_mask
        elif model.task == '2':
            target = batch['ans'].numpy()
            valid_mask = batch['valid_mask'].numpy()
            test_mask = batch['test_mask'].numpy()
            validation_flag = (1-valid_mask)*test_mask
            training_flag = test_mask*valid_mask
        loss.backward()
        optimizer.step()
        all_preds.append(output[training_flag == 1])
        all_targets.append(target[training_flag == 1])
        val_preds.append(output[validation_flag == 1])
        val_targets.append(target[validation_flag == 1])
        train_loss += float(loss.detach().cpu().numpy())
        batch_idx += 1

    all_pred = np.concatenate(all_preds, axis=0)
    all_target = np.concatenate(all_targets, axis=0)
    val_pred = np.concatenate(val_preds, axis=0)
    val_target = np.concatenate(val_targets, axis=0)
    #model.eval()
    if model.task == '1':
        train_auc = compute_auc(all_target, all_pred)
        mlflow.log_metric("train_auc",train_auc)
        val_auc = compute_auc(val_target, val_pred)
        mlflow.log_metric("val_auc",val_auc)
        train_accuracy = compute_accuracy(all_target, all_pred)
        mlflow.log_metric("train_accuracy",train_accuracy)

        val_accuracy = compute_accuracy(val_target, val_pred)
        mlflow.log_metric("val_accuracy",train_accuracy)

        # log the model
        # Create model signature
        signature = infer_signature(all_pred, all_target)
        # log it
        mlflow.pytorch.log_model(model,artifact_path="model",signature=signature)
        mlflow.pytorch.log_state_dict(model.state_dict(),artifact_path="model")
    if model.task == '2':
        raise NotImplemented("Only imnplemented task 1")
    if best_acc is None or val_accuracy > best_acc:
        best_acc = val_accuracy
        best_epoch = epoch
        best_run = mlflow.active_run().info.run_id


def run(experiment_id,model_parameters:dict, run_parameters:dict):
    '''Runs the task as a function call instead of from the CLI. This function
    logs data as appropriate to the MLFlow server.
    '''
    global best_acc, best_epoch, best_run

    with mlflow.start_run(experiment_id=experiment_id,nested=True):
        # log parameters
        mlflow.log_params(run_parameters)
        mlflow.log_params(model_parameters)
        
        # set up the run parameters
        np.random.seed(run_parameters["seed"])
        torch.backends.cudnn.deterministic = run_parameters["deterministic"]
        torch.backends.cudnn.benchmark = run_parameters["benchmark"]
        torch.manual_seed(run_parameters["seed"])
        np.random.seed(run_parameters["seed"])

        if model_parameters["is_dash"]==1:
            answer_filename = 'data_task_1_2/answer_dash_metadata_task_1_2_extra.json'
            answer_meta = open_json(answer_filename)
        else:
            answer_meta = None

        train_data = open_json('data_task_1_2/data_1_2.json')
        for d in train_data:
            d['valid_mask'] = [0 if np.random.rand(
            ) < run_parameters["valid_prob"] and ds else 1 for ds in d['test_mask']]

        train_dataset = LSTMDataset(train_data, answer_meta=answer_meta)
        collate_fn = lstm_collate(model_parameters["is_dash"] == 1)
        num_workers = 2
        bs = run_parameters["batch_size"]
        train_loader = torch.utils.data.DataLoader(
            train_dataset, collate_fn=collate_fn, batch_size=bs, num_workers=num_workers, shuffle=True, drop_last=False)

        if run_parameters["model"] == 'lstm':
            model = LSTMModel(**model_parameters).to(device)
            # model = LSTMModel(n_question=27613, n_user=118971, n_subject=389, task=params.task, s_dim=params.question_dim,
            #                 n_quiz=17305, n_group=11844, is_dash=params.dash == 1, hidden_dim=params.hidden_dim, q_dim=params.question_dim, dropout=params.dropout, default_dim=params.default_dim, bidirectional=params.bidirectional).to(device)
        elif run_parameters["model"] == 'attn':
            raise NotImplemented("Haven't done attn yet")
            # model = AttentionModel(n_question=27613, n_user=118971, n_subject=389, task=params.task, s_dim=params.question_dim,
            #                     n_quiz=17305, n_group=11844, is_dash=params.dash == 1, hidden_dim=params.hidden_dim, q_dim=params.question_dim, dropout=params.dropout,  default_dim=params.default_dim).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=run_parameters["lr"], weight_decay=run_parameters["weight_decay"])


        for epoch in range(run_parameters["max_epochs"]):
            with mlflow.start_run(experiment_id=experiment_id,nested=True):
                train_model(model,train_loader,optimizer,epoch)
                # exit condition is ten less performant epochs
                if (epoch-best_epoch) > 10:
                    break
        
        # identify the best run, as determined by the authors
        mlflow.log_param("best_run",best_run)
        mlflow.log_param("best_run_name",mlflow.get_run(best_run).info.run_name)
        mlflow.log_param("best_run_uri",mlflow.get_run(best_run).info.artifact_uri)