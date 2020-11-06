# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:04:50 2020

@author: Daniel Tan
"""

import os
import torch
import pickle
from torch.utils.data import Dataset
from problems.dtspms.state_dtspms import StateDTSPMS

class DTSPMS(object):
    
    NAME = 'dtspms'
    
    @staticmethod
    def get_costs(dataset, pi):
        """
        This function 
        
        :param dataset:     A DTSPMS instance
        :param pi:          torch.Tensor (B, N, 1) (I think?)
        """
        print(f"Pi shape: {pi.shape}")
        assert DTSPMS._is_valid(pi), "Invalid solution"
    
        loc = torch.cat([
            dataset['pickup_depot'][:,None,:],
            dataset['pickup_loc'],
            dataset['dropoff_depot'][:,None,:],
            dataset['dropoff_loc']
        ], dim=1)
        
        total_items = dataset['pickup_loc'].size(1)
        
        # Filter out the stack actions
        pi = pi[pi < 2 * total_items + 2]
    
        d = loc.gather(1, pi.unsqueeze(-1).expand_as(dataset))
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            # In the transit phase, we move from the pickup to dropoff depot
            # but this should not be counted for the sake of evaluating the policy
            - (dataset['pickup_depot'] - dataset['dropoff_depot']).norm(p=2, dim=1) # Pickup to dropoff deopt, not counted
        ), None
    
    @staticmethod
    def make_dataset(*args, **kwargs):
        return DTSPMSDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateDTSPMS.initialize(*args, **kwargs)
    
    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        raise NotImplementedError
    
    @staticmethod
    def is_valid(pi):
        # Each sequence should be a permutation of (0, 1, ..., 2N+1)
        return (torch.arange(pi.size(1), out=pi.data.new()) \
                     .view(1, -1).expand_as(pi) \
            == pi.data.sort(1)[0]).all()
    
class DTSPMSDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000,
                 offset=0, distribution=None, num_stacks = 2, stack_size = 16):
        super(DTSPMSDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            # Not yet sure how to load a dataset
            raise NotImplementedError
            
            # Legacy code
            assert os.path.splitext(filename)[1] == '.pkl'
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            indices = torch.arange(1, size+1).to(torch.float32).view(-1,1)
            
            self.data = [
                {
                    # Non-depot nodes are augmented with :
                    # - index of the item they contain
                    # - whether they are pickup or dropoff
                    
                    # This is necessary because the same embedder
                    # is used for all such nodes
                    'pickup_loc': torch.cat((
                        torch.FloatTensor(size, 2).uniform_(0, 1),
                        indices,
                        torch.zeros((size, 0))
                    ), 1),
                    'dropoff_loc': torch.cat((
                        torch.FloatTensor(size, 2).uniform_(0,1),
                        indices, 
                        torch.zeros((size, 1))
                    ), 1),
                    'pickup_depot': torch.FloatTensor(2).uniform_(0, 1),
                    'dropoff_depot': torch.FloatTensor(2).uniform(0, 1),
                    'stack_size': stack_size, 
                    'num_stacks': num_stacks,
                }
                for i in range(num_samples)
            ]
        self.size = num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
