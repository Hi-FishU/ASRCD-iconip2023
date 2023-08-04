
from argparse import ArgumentParser
import torch
import numpy as np


class CommonArgParser(ArgumentParser):
    def __init__(self) -> None:
        super(CommonArgParser, self).__init__()
        self.add_argument('--exer_n', type=int, default=798,
                          help='The number for exercise.')
        self.add_argument('--knowledge_n', type=int, default=95,
                          help='The number for knowledge concept.')
        self.add_argument('--student_n', type=int, default=5254,
                          help='The number for student.')
        self.add_argument('--gpu', type=int, default=1,
                          help='The id of gpu, e.g. 0.')
        self.add_argument('--epoch_n', type=int, default=3,
                          help='The epoch number of training')
        self.add_argument('--lr', type=float, default=0.0001,
                          help='Learning rate')
        self.add_argument('--dropout', type=float, default=0.1,
                          help='Dropout rate')
        self.add_argument('--eps', type=float, default=1e-8,
                          help='Epsilon')
        self.add_argument('--l2', type=float, default=0,
                          help='L2 decay weight')
        self.add_argument('--hidden_1', type=int, default=128,
                         help='1st FCL width')
        self.add_argument('--hidden_2', type=int, default=256,
                          help='2nd FCL width')
        self.add_argument('--hidden_3', type=int, default=512,
                          help='3rd FCL width')
        self.add_argument('--hidden_4', type=int, default=64,
                          help='4th FCL width')
        self.add_argument('--hidden_5', type=int, default=16,
                          help='5th FCL width')
        self.add_argument('--batch_size', type=int, default=64,
                          help='Batch size')
        self.add_argument('--epoch', type=int, default=10,
                          help='Epoch num')
        self.add_argument('--isDeep', type=bool, default=False,
                          help='Use deeper FCN')
        self.add_argument('--isRes', type=bool, default=False,
                          help='Use Res Net')