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
from problems.dtspms.state_dtspms import StateDTSPMS

class DTSPMS(object):
    
    NAME = 'dtspms'
    
    @staticmethod
    def get_costs(dataset, pi):
        """
        Get the cost associated with a particular solution 
        
        :param dataset:     A DTSPMS instance
        :param pi:          torch.Tensor (B, sequence_len)
                            A valid solution. No error checking is done!
        """
    
        loc = torch.cat([
            dataset['pickup_depot'][:,None,:],
            dataset['pickup_loc'],
            dataset['dropoff_depot'][:,None,:],
            dataset['dropoff_loc']
        ], dim=1)
        
        total_items = dataset['pickup_loc'].size(1)
        
        # Filter out the stack actions
        batch_size = pi.size(0)
        # The policy includes travel to stack nodes, but we want only the item nodes
        pi = pi[pi < 2 * total_items + 2]
        pi = pi.view(batch_size,-1)
        index = pi[..., None].expand(*pi.size(), loc.size(-1))
        d = loc.gather(1, index)
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
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = DTSPMS.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)
    
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
            
            self.data = [
                {
                    # Convention in the literature is pickup / delivery depot at (50,50)
                    # all locations sampled uniformly in (0, 100)
                    'pickup_loc': torch.FloatTensor(size, 2).uniform_(0, 100),
                    'dropoff_loc': torch.FloatTensor(size, 2).uniform_(0,100),
                    'pickup_depot': torch.Tensor([50.0, 50.0]).to(torch.float32),
                    'dropoff_depot': torch.Tensor([50.0, 50.0]).to(torch.float32),
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
