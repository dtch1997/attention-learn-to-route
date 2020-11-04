# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:04:50 2020

@author: Daniel Tan
"""

import os
import torch
import pickle
from torch.utils.data import Dataset
from problems.dtspms.state_dtsppl import StateDTSPMS

class DTSPMS(object):
    
    NAME = 'dtsppl'
    def __init__(self, reload_cost = 0.1, reload_depth = 1):
        """
        :param reload_cost:     Cost of reloading an item once
        :param reload_depth:    Max items that can be unloaded at any point in time
        """
        self.reload_cost = reload_cost
        self.reload_depth = reload_depth
    
    @staticmethod
    def get_costs(dataset, pi):
        """
        :param dataset:     A DTSPPL instance
        :param pi:          torch.Tensor (B, N, 1) (I think?)
        """
        print(f"Pi shape: {pi.shape}")
        assert DTSPMS._is_valid(pi), "Invalid solution"
        
    
    @staticmethod
    def make_dataset(*args, **kwargs):
        return DTSPMSDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateDTSPMS.initialize(*args, **kwargs)
    
    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        pass
    
    @staticmethod
    def is_valid(pi):
        pass
    
class DTSPMSDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000,
                 offset=0, distribution=None):
        super(DTSPMSDataset, self).__init__()

        if filename is not None:
            # Not yet sure how to load a dataset
            raise NotImplementedError
            
            # Legacy code
            assert os.path.splitext(filename)[1] == '.pkl'
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # Sample points randomly in [0, 1] square for pickup and delivery
            self.pickup_data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]
            self.dropoff_data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Sample a problem instance
        return self.pickup_data[idx], self.dropoff_data[idx]
