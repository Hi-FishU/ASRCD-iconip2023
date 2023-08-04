import time

import numpy as np
import pandas as pd
import torch
from torch._C import import_ir_module
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from sklearn.metrics import mean_squared_error, roc_auc_score

from dataload import dataset_load
from models import GCD_Res_MLP as Res_MLP

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def evaluation(
    model,
    data_loader,
    difficulty,
    student_n,
    skill_k,
    exer_e,
    hidden_dim_1=512,
    hidden_dim_2=256,
    res_mode=True,
    is_deep=True,
    graph='None',
    dropout=0.3,
):
    correct, total = 0., 0
    all_preds = []
    all_labels = []
    # model_eval = Res_MLP(student_n=student_n,
    #                      skill_k=skill_k,
    #                      exer_e=exer_e,
    #                      hidden_dim_1=hidden_dim_1,
    #                      hidden_dim_2=hidden_dim_2,
    #                      dropout=dropout,
    #                      res_mode=res_mode,
    #                      is_deep=is_deep,
    #                      relation_type=graph,
    #                      device=device)
    # model_eval.load_state_dict(model.state_dict())
    # model_eval = model_eval.to(device)
    model_eval = model.eval()
    for index, data in enumerate(data_loader):
        # Categogrical encoding
        skill, stu_id, exer_id, labels = data
        output = model_eval(stu_id, exer_id, skill, difficulty)

        pred = torch.as_tensor(output.squeeze()) > 0.5
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(output.squeeze().detach().cpu().numpy().tolist())

        labels = torch.as_tensor(labels) == 1
        labels = labels.to(device)
        correct += torch.eq(pred, labels).sum()
        total += len(labels)

    auc = roc_auc_score(all_labels, all_preds)
    rmse = mean_squared_error(all_labels, all_preds)**0.5
    acc = correct.item() / len(all_labels)
    return acc, auc, rmse


def train_weight(student_n,
                 skill_k,
                 exer_e,
                 hidden_dim_1=512,
                 hidden_dim_2=256,
                 dropout=0.3,
                 epochs=5,
                 lr=0.005,
                 lr_decay=5,
                 gamma=0.5,
                 res_mode=True,
                 is_deep=True,
                 graph='Chi',
                 batch_size=32,
                 train_ratio=0.6,
                 val_ratio=0.3,
                 eps=1e-8,
                 sche_milestones=[],
                 L2_lamb=0.0,
                 device=None):
    """Training Weight Parameter

    Args:
        student_n (int): Amount of student
        skill_k (int): Amount of knowledge concept
        exer_e (int): Amount of exercise
        hidden_dim_1 (int, optional): Num of the first layer of FNN. Defaults to 512.
        hidden_dim_2 (int, optional): Num of the second layer of FNN. Defaults to 256.
        dropout (float, optional): Dropout rate of FNN. Defaults to 0.3.
        epochs (int, optional): Epoch num of training. Defaults to 5.
        lr (float, optional): Learning rate of training. Defaults to 0.005.
        lr_decay (int, optional): After epoch(es) operate the lr scheduler(only available for SterLR scheduler). Defaults to 5.
        gamma (float, optional): Lr decay rate. Defaults to 0.5.
        res_mode (bool, optional): Whether using the ResNet. Defaults to True.
        is_deep (bool, optional): Whether using the deeper FNN (5 layers). Defaults to True.
        graph (str, optional): Relation type of the knowledge concepts, included 'Chi', 'Dense', 'Transition', 'None', 'MHA', 'PAM'. Defaults to 'Chi'.
        batch_size (int, optional): Batch size of the dataset. Defaults to 32.
        train_ratio (float, optional): Ratio of train dataset. Defaults to 0.6.
        val_ratio (float, optional): Ratio of validation dataset. Defaults to 0.3.
        eps (float, optional): Epsilon for Adam optimizer. Defaults to 1e-8.
        sche_milestones (list, optional): Decay epoch for the MultiStepLR. Defaults to [].
        L2_lamb (float, optional): L2 weight for Adam optimizer. Defaults to 0.0.
        device ([type], optional): Device of training. Defaults to None.

    Returns:
        Metric: Training score
    """
    assert graph in [
        'Chi', 'Dense', 'Transition', 'None', 'MHA', 'PAM', 'Chi_MHA',
        'Chi_PAM'
    ]
    train_data_loader, valid_data_loader, test_data_loader = dataset_load(
        graph=graph,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        shuffle=True)
    record_time = str(
        time.strftime("%Y%m%d %H%M%S", time.localtime(time.time())))

    model = Res_MLP(student_n=student_n,
                    skill_k=skill_k,
                    exer_e=exer_e,
                    hidden_dim_1=hidden_dim_1,
                    hidden_dim_2=hidden_dim_2,
                    dropout=dropout,
                    res_mode=res_mode,
                    is_deep=is_deep,
                    relation_type=graph,
                    device=device)
    model = model.to(device)
    #criterion = nn.BCELoss()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           eps=eps,
                           weight_decay=L2_lamb)
    # scheduler = optim.lr_scheduler.StepLR(optimizer,
    #                                       step_size=lr_decay,
    #                                       gamma=gamma)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                  mode='min',
    #                                                  patience=30,
    #                                                  factor=0.5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=sche_milestones,
                                               gamma=gamma)

    with open(r'temp/skill_correct.npy', mode='rb') as f:
        difficulty = 1 - np.load(f)[:, 2]
    #Parameter record
    with open(r'model/log/Record_{}.txt'.format(record_time), mode='a') as f:
        f.write(
            'Config:\n\tgraph:{}\tbatch size:{}\tlr:{}\tl2:{}\teps:{}\tepoch:{}\tRes:{}\tDeep:{}\n\nTrain_Record:\n'
            .format(graph, batch_size, lr, L2_lamb, eps, epochs, res_mode,
                    is_deep))

    for epoch in range(epochs):
        correct, total = 0, 0
        epoch_loss = []
        print('Current lr:{}'.format(
            optimizer.state_dict()['param_groups'][0]['lr']))

        for data in tqdm.tqdm(train_data_loader):
            model.train()
            skill, stu_id, exer_id, labels = data
            optimizer.zero_grad()

            output = model(stu_id, exer_id, skill, difficulty)
            labels = labels.to(device)
            labels_mask = nn.functional.one_hot(labels, num_classes=2)
            output1 = torch.ones(output.size()).to(device) - output
            outputs = torch.cat((output1, output), 1)

            #NLLLoss
            loss = criterion(torch.log(outputs), labels)
            #BCELoss
            #loss = criterion(outputs.float(), labels_mask.float())

            epoch_loss.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
            model.apply_clipper()

        epoch_acc, epoch_auc, _ = evaluation(student_n=student_n,
                                             skill_k=skill_k,
                                             exer_e=exer_e,
                                             model=model,
                                             hidden_dim_1=hidden_dim_1,
                                             hidden_dim_2=hidden_dim_2,
                                             data_loader=train_data_loader,
                                             res_mode=res_mode,
                                             is_deep=is_deep,
                                             graph=graph,
                                             difficulty=difficulty)
        acc_valid, auc_valid, rmse_valid = evaluation(
            student_n=student_n,
            skill_k=skill_k,
            exer_e=exer_e,
            model=model,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            data_loader=valid_data_loader,
            res_mode=res_mode,
            is_deep=is_deep,
            graph=graph,
            difficulty=difficulty)

        print('Train loss for epoch {}: {}, accuracy: {}, AUC:{}'.format(
            epoch + 1, np.mean(epoch_loss), epoch_acc, epoch_auc))
        print('Valid RMSE for epoch {}: {}, accuracy: {}, AUC:{}'.format(
            epoch + 1, rmse_valid, acc_valid, auc_valid))

        with open(r'model/log/Record_{}.txt'.format(record_time),
                  mode='a') as f:
            f.write('Current lr:{}\n'.format(
                optimizer.state_dict()['param_groups'][0]['lr']))
            f.write(
                'Train loss for epoch {}: {}, accuracy: {}, AUC:{}\n'.format(
                    epoch + 1, np.mean(epoch_loss), epoch_acc, epoch_auc))
            f.write(
                'Valid RMSE for epoch {}: {}, accuracy: {}, AUC:{}\n'.format(
                    epoch + 1, rmse_valid, acc_valid, auc_valid))

        scheduler.step()
    # Eval
    model.eval()

    correct, total = 0., 0
    all_preds = []
    all_labels = []

    acc_train, auc_train, rmse_train = evaluation(
        student_n=student_n,
        skill_k=skill_k,
        exer_e=exer_e,
        model=model,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        data_loader=train_data_loader,
        res_mode=res_mode,
        is_deep=is_deep,
        graph=graph,
        difficulty=difficulty)

    stu_knowledge_status = {}
    interaction_emb = []
    for index, data in enumerate(test_data_loader):
        # Categogrical encoding
        skill, stu_id, exer_id, label = data
        output = model(stu_id, exer_id, skill, difficulty)

        pred = torch.as_tensor(output.squeeze()) > 0.5
        all_labels.extend(labels.cpu().detach().numpy().tolist())
        all_preds.extend(output.squeeze().detach().cpu().numpy().tolist())
        labels = torch.as_tensor(label) == 1
        labels = labels.to(device)
        correct += torch.eq(pred, labels).sum()
        total += len(labels)
        stu_knowledge_status.update(
            dict(
                zip(
                    stu_id.cpu().numpy().tolist(),
                    model.get_knowledge_status(
                        stu_id.to(device)).cpu().numpy().tolist())))
        interaction_emb.extend(
            np.concatenate([
                model.get_interaction(stu_id, exer_id, skill,
                                      difficulty).cpu().detach().numpy(),
                label.unsqueeze(0).numpy().T
            ], 1).tolist())

    auc = roc_auc_score(all_labels, all_preds)
    rmse = mean_squared_error(all_labels, all_preds)**0.5
    acc = correct.item() / len(all_labels)
    with open(r'model/log/Record_{}.txt'.format(record_time), mode='a') as f:
        f.write(
            'TRAIN:\n-------------------------------------------------------------\n'
        )
        f.write('|\tAccuracy: {:.8f}, AUC: {:.8f}, RMSE: {:.8f}\t|\n'.format(
            acc_train, auc_train, rmse_train))
        f.write(
            '-------------------------------------------------------------\n')
        f.write(
            'TEST:\n-------------------------------------------------------------\n'
        )
        f.write('|\tAccuracy: {:.8f}, AUC: {:.8f}, RMSE: {:.8f}\t|\n'.format(
            acc, auc, rmse))
        f.write(
            '-------------------------------------------------------------\n')
    print(
        'TRAIN:\n-----------------------------------------------------------------\n|\t',
        end='')
    print('Accuracy: {:.8f}, AUC: {:.8f}, RMSE: {:.8f}\t|'.format(
        acc_train, auc_train, rmse_train))
    print('-----------------------------------------------------------------')
    print(
        'TEST:\n-----------------------------------------------------------------\n|\t',
        end='')
    print('Accuracy: {:.8f}, AUC: {:.8f}, RMSE: {:.8f}\t|'.format(
        acc, auc, rmse))
    print('-----------------------------------------------------------------')
    torch.save(model.state_dict(),
               'model/Res_MLP_ep{}_{:.3f}.pt'.format(epochs, auc))
    print('Extracting students knowledge status...')
    pd.DataFrame.from_dict(
        stu_knowledge_status, orient='index',
        columns=skill_id).to_csv('student_knowledge_status.csv')
    print('Extracting interaction embedding...')
    pd.DataFrame(interaction_emb).to_csv('interaction.csv')
    return acc, auc, rmse, acc_train, auc_train, rmse_train


if __name__ == '__main__':
    with open(r'temp/skill_id.npy', mode='rb') as f:
        skill_id = np.load(f)
    with open(r'temp/stu_id.npy', mode='rb') as f:
        stu_id = np.load(f)
    with open(r'temp/exer_id.npy', mode='rb') as f:
        exer_id = np.load(f)
    print(
        'Dataset Summary:\n\tStudent Count:{}\n\tSkill Count:{}\n\tExercise Count:{}'
        .format(len(stu_id), len(skill_id), len(exer_id)))
    train_weight(student_n=max(stu_id),
                 skill_k=len(skill_id),
                 exer_e=max(exer_id),
                 hidden_dim_1=512,
                 hidden_dim_2=256,
                 dropout=0.3,
                 epochs=5,
                 lr=0.001,
                 eps=3e-8,
                 L2_lamb=1e-4,
                 lr_decay=8,
                 graph='Chi_MHA',
                 res_mode=True,
                 is_deep=True,
                 sche_milestones=[5, 8],
                 batch_size=256,
                 device=device)
    print('done')