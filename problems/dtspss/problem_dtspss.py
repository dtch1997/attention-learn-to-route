# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:04:50 2020

@author: Daniel Tan
"""

import os
import torch
import pickle
from torch.utils.data import Dataset
from utils.beam_search import beam_search

from problems.dtspss.state_dtspss import StateDTSPSS

class DTSPSS(object):
    
    NAME = 'dtspss'
    
    @staticmethod
    def get_costs(dataset, pi):
        """
        :param dataset:     A DTSPMS instance
        :param pi:          torch.Tensor (B, N, 1) (I think?)
        """
        print(f"Pi shape: {pi.shape}")
        assert DTSPSS._is_valid(pi), "Invalid solution"
    
        loc = torch.cat([
            dataset['pickup_depot'][:,None,:],
            dataset['pickup_loc'],
            dataset['dropoff_depot'][:,None,:],
            dataset['dropoff_loc']
        ], dim=1)
    
        d = loc.gather(1, pi.unsqueeze(-1).expand_as(dataset))
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            # + (d[:, 0] - dataset['pickup_depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['dropoff_depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
            - (dataset['pickup_depot'] - dataset['dropoff_depot']).norm(p=2, dim=1) # Pickup to dropoff deopt, not counted
        ), None
    
    @staticmethod
    def make_dataset(*args, **kwargs):
        return DTSPSSDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateDTSPSS.initialize(*args, **kwargs)
    
    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = DTSPSS.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)
    
    @staticmethod
    def is_valid(pi):
        # Each sequence should be a permutation of (0, 1, ..., 2N+1)
        return (torch.arange(pi.size(1), out=pi.data.new()) \
                     .view(1, -1).expand_as(pi) \
            == pi.data.sort(1)[0]).all()
    
class DTSPSSDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000,
                 offset=0, distribution=None):
        super(DTSPSSDataset, self).__init__()

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
            self.data = [
                {
                    'pickup_loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    'dropoff_loc': torch.FloatTensor(size, 2).uniform_(0,1),
                    'pickup_depot': torch.FloatTensor(2).uniform_(0, 1),
                    'dropoff_depot': torch.FloatTensor(2).uniform_(0, 1),
                }
                for i in range(num_samples)
            ]
        self.size = num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
