#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：train_code
@File    ：data.py
@Author  ：liangliang yan
@Date    ：2022/3/4 15:13
"""


import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import h5py
import os
from torch.utils.data import Dataset, DataLoader
from torch import optim
from visdom import Visdom


class ReadHDF5File:
    """This class is to fetch the corresponding physical data"""
    pass


class GenerateDataset:
    """This class is to generate datasets for hdf5 files"""
    def __init__(self, path, trainDATA_testDATA_Boundary=None):
        self.file_list = None
        self.trainDATA_testDATA_Boundary = trainDATA_testDATA_Boundary
        self.file_path = path
        self.key = None

    def get_File_List(self):
        """
        get all of the data's name(but it have the postfix)
        :return: list
        """
        files = os.listdir(self.file_path)
        files.sort()
        self.file_list = files
        return self.file_list

    def get_File_Path(self, file_name):
        """
        get all of the data's path
        :param file_name: string
        :return: string
        """
        return os.path.join(self.file_path + '\\' + file_name)

    def get_File_Key(self, filename):
        """
        get the key value of the group of the file
        :return: list
        """
        path = os.path.join(self.file_path+'\\'+filename)
        f = h5py.File(path)
        key = []
        for i in f:
            key.append(i)
        self.key = key
        return key

    def get_trainDATA_testDATA_Boundary(self, MAX):
        """
        get the boundary of the training data and test data
        :param MAX: int
        :return: int
        """
        self.trainDATA_testDATA_Boundary = MAX
        return self.trainDATA_testDATA_Boundary

    def __getitem__(self, item):
        """
        get the data's path though the index
        :param item: int
        :return: string
        """
        return os.path.join(self.file_path+'\\'+self.file_list[item])

    def __len__(self):
        """
        get the length of this file
        :return: int
        """
        return len(self.file_list)

    def get_Dataset(self):
        """
        this function is to split the train data and test data
        :return:list
        """
        assert self.trainDATA_testDATA_Boundary is not None
        assert self.key is not None
        DATA_train = []
        DATA_test = []
        i = 0

        for file in self.file_list:
            filename = os.path.splitext(file)
            if filename[1] == '.hdf5':
                path = os.path.join(self.file_path+'\\'+file)
                f = h5py.File(path)
                data_ = f[self.key[0]]
                # print(data_)
                if i <= self.trainDATA_testDATA_Boundary:
                    DATA_train.append(data_)
                else:
                    DATA_test.append(data_)
                i += 1
        return DATA_train, DATA_test

    def select_Label_As_Input(self):
        pass

    def select_Label_As_Output(self):
        pass


def trans_To_Arr(DATA, type_):
    return np.array(DATA, dtype=type_)


if __name__ == '__main__':
    data = GenerateDataset('..\\DATA')
    print(data.get_File_List()[0])
    print(len(data))
    print(data.get_File_Path("snapshot_097.hdf5"))
    print(data[0])
    print(*data.get_File_Key('snapshot_097.hdf5'))
    data.get_trainDATA_testDATA_Boundary(1)
    data1, data2 = data.get_Dataset()
    # data1 = trans_To_Arr(data1, np.float32)
    print(data1, len(data2))



