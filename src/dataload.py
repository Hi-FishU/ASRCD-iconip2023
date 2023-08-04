import sys

sys.path.append("..")

import torch
import numpy as np
from models import relevance_test as rt
from torch.utils.data import DataLoader, Dataset
from graph import *


class GCDDataSet(Dataset):
    def __init__(self, skills, stu_id, exer_id, answers):
        super(GCDDataSet, self).__init__()

        #block the skill of exercise

        self.skills = skills
        self.stu_id = stu_id
        self.exer_id = exer_id
        self.answers = answers

    def __getitem__(self, index):
        return self.skills[index], self.stu_id[index], self.exer_id[
            index], self.answers[index]

    def __len__(self):
        return len(self.answers)


class ASRCDDataSet(Dataset):
    def __init__(self, skills, stu_id, exer_id, answers):
        super(ASRCDDataSet, self).__init__()

        #block the skill of exercise
        split_index = len(skills[0]) / 2
        self.skills = skills[:, int(split_index):]
        self.stu_id = stu_id
        self.exer_id = exer_id
        self.answers = answers

    def __getitem__(self, index):
        return self.skills[index], self.stu_id[index], self.exer_id[
            index], self.answers[index]

    def __len__(self):
        return len(self.answers)


def relevance(p=0.005):
    with open(r'/home/fishu/IGCD/temp/stu_skill.npy', mode='rb') as f:
        stu_skill = np.load(f)
    cov = rt(stu_skill, p=p)
    with open(r'/home/fishu/IGCD/temp/cov.npy', mode='wb') as f:
        np.save(f, cov)
    return cov


def dataset_load(graph='Dense',
                 batch_size=256,
                 train_ratio=0.6,
                 val_ratio=0.2,
                 shuffle=True):

    #Temporary files Load
    print('Loading the DataSet...')
    with open(r'/home/fishu/IGCD/temp/exer_id.npy', mode='rb') as f:
        exer_id = np.load(f)
    with open(r'/home/fishu/IGCD/temp/stu_id.npy', mode='rb') as f:
        stu_id = np.load(f)
    with open(r'/home/fishu/IGCD/temp/stu_answer.npy', mode='rb') as f:
        stu_answer = np.load(f)
    with open(r'/home/fishu/IGCD/temp/stu_skill.npy', mode='rb') as f:
        stu_skill = np.load(f)
    with open(r'/home/fishu/IGCD/temp/q_matrix.npy', mode='rb') as f:
        q_matrix = np.load(f)

    #extend Q_matrix
    print('Calculate the Knowledge Graph...')
    q_matrix_extend = np.zeros(q_matrix.shape, dtype=int)

    if graph in ['Chi', 'Chi_MHA', 'Chi_PAM']:
        try:
            with open(r'/home/fishu/IGCD/temp/chi.npy', mode='rb') as f:
                cov = np.load(f)
        except:
            cov = build_chi_graph(p=0.005)
    elif graph == 'Dense':
        try:
            with open(r'/home/fishu/IGCD/temp/dense.npy', mode='rb') as f:
                cov = np.load(f)
        except:
            cov = build_dense_graph(stu_skill.shape[1])
    elif graph == 'Transition':
        try:
            with open(r'/home/fishu/IGCD/temp/transition.npy', mode='rb') as f:
                cov = np.load(f)
        except:
            cov = build_transition_matrix(stu_skill.shape[1])

    for i in range(0, len(q_matrix_extend)):
        if graph not in ['Chi', 'Dense', 'Transition', 'Chi_MHA', 'Chi_PAM']:
            q_matrix_extend = q_matrix
            break
        corr_i = np.zeros(len(q_matrix_extend[i]), dtype=int)
        for j in range(0, len(q_matrix_extend[i])):
            if q_matrix[i][j]:
                corr_i = cov[j] + corr_i
        corr_i = corr_i + q_matrix[i]
        corr_i = np.array(np.where(corr_i >= 1, 1, 0), dtype=int)
        q_matrix_extend[i] = corr_i
    print('Extracting the Knowledge Graph...')
    # with open(r'Knowledge Graph.csv', mode='w') as f:
    #     np.savetxt(f, cov, fmt='%1.1f')
    np.save(r'/home/fishu/IGCD/temp/q_matrix_ext.npy', q_matrix_extend)
    print('Original KCE: {:.3f} concepts/exer'.format(
        np.mean(np.sum(q_matrix, axis=1))))
    if graph in ['Chi', 'Dense', 'Transition', 'Chi_MHA', 'Chi_PAM']:
        print('After Extend: {:.3f} concepts/exer'.format(
            np.mean(np.sum(q_matrix_extend, axis=1))))
    #student skill
    print('Setting the DataLoader...')
    index = []
    for i in range(0, len(stu_answer)):
        for j in range(0, len(stu_answer[i])):
            if stu_answer[i][j]:
                index.append((i, j))

    #Array construction
    input_stu_id = np.zeros(len(index), dtype=np.int64)
    input_exer_id = np.zeros(len(index), dtype=np.int64)
    answers = np.zeros(len(index))
    skills = np.zeros((len(index), 2 * q_matrix.shape[1]), dtype=np.int64)

    for i in range(0, len(index)):
        input_stu_id[i] = stu_id[index[i][0]]
        input_exer_id[i] = exer_id[index[i][1]]
        answers[i] = stu_answer[index[i][0]][index[i][1]]
        skills[i] = np.concatenate(
            [q_matrix_extend[index[i][1]], q_matrix[index[i][1]]])
    answers = np.array(np.where(answers < 0, 0, 1), dtype=np.int64)
    #dataloader
    asrcd_dataset = ASRCDDataSet(skills, input_stu_id, input_exer_id, answers)
    amount = len(input_stu_id)
    train_size = int(train_ratio * amount)
    val_size = int(val_ratio * amount)
    test_size = amount - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        asrcd_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(3619))

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   drop_last=True)
    valid_data_loader = DataLoader(val_dataset,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   drop_last=True)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  drop_last=True)

    return train_data_loader, valid_data_loader, test_data_loader


if __name__ == '__main__':
    dataset_load(graph='Dense')
    dataset_load(graph='Chi_MHA')
    dataset_load(graph='Transition')
