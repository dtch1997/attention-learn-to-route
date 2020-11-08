# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 00:38:48 2020

@author: Daniel Tan
"""

import torch
import unittest
from problems.dtspms.state_dtspms import StateDTSPMS
from problems.dtspms.problem_dtspms import DTSPMS, DTSPMSDataset   
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class TestAttentionModel(unittest.TestCase):
    """Test the AttentionModel on the DTSPMS"""    
    def test_forward_pass(self):
        from nets.attention_model import AttentionModel, set_decode_type
        from reinforce_baselines import NoBaseline
        from torch.utils.data import DataLoader
        
        model = AttentionModel(
            embedding_dim = 8,
            hidden_dim = 8,
            problem = DTSPMS,
            n_encode_layers=3,
            mask_inner=True,
            mask_logits=True,
            normalization='batch',
            tanh_clipping=10,
            checkpoint_encoder=True,
            shrink_size=None
        ).to(torch.device('cpu'))
        
        model.train()
        set_decode_type(model, "sampling")
        
        baseline = NoBaseline()
        training_dataset = baseline.wrap_dataset(model.problem.make_dataset(
            size=5, num_samples=10, distribution=None))
        training_dataloader = DataLoader(training_dataset, batch_size=2, num_workers=0)

        for batch in training_dataloader:
            cost, ll = model(batch)
            break
    
    def test_partial_forward_pass(self):
        from nets.attention_model import AttentionModel, set_decode_type
        from reinforce_baselines import NoBaseline
        from torch.utils.data import DataLoader
        
        model = AttentionModel(
            embedding_dim = 8,
            hidden_dim = 8,
            problem = DTSPMS,
            n_encode_layers=3,
            mask_inner=True,
            mask_logits=True,
            normalization='batch',
            tanh_clipping=10,
            checkpoint_encoder=True,
            shrink_size=None
        ).to(torch.device('cpu'))
        
        model.train()
        set_decode_type(model, "sampling")
        
        baseline = NoBaseline()
        training_dataset = baseline.wrap_dataset(model.problem.make_dataset(
            size=5, num_samples=10, distribution=None))
        training_dataloader = DataLoader(training_dataset, batch_size=2, num_workers=0)

        for batch in training_dataloader:
            init_embed = model._init_embed(batch)
            state = model.problem.make_state(batch)
            self.assertTrue(init_embed.size() == (2,14,8))
            final_embed, _ = model.embedder(init_embed)
            self.assertTrue(final_embed.size() == (2,14, 8))
            context = model._get_parallel_step_context(final_embed, state)
            self.assertTrue(context.size() == (2,1,24))
            _log_p, pi = model._inner(batch, final_embed)
            cost, mask = model.problem.get_costs(batch, pi)
            break
        


class TestDTSPMS(unittest.TestCase):
    """Unit test for the DTSPMS problem"""
    def test_get_cost(self):
        # TODO: Implement
        pass

class TestDTSPMSDataset(unittest.TestCase):
    def test_initialize_new(self):
        """Test initializing the dataset"""
        dataset = DTSPMSDataset(size = 5, num_samples = 32, num_stacks=2, stack_size=5)    
    
    def test_load_generated(self):
        dataset = DTSPMSDataset(filename = 'data/dtspms/dtspms20_validation_seed4321.pkl')        
    
class TestStateDTSPMS(unittest.TestCase):
    def test_dry_run(self):
        """Dry run a valid solution. Test passes as long as the solution is valid"""
        input = {
            'pickup_depot': torch.FloatTensor([0, 0]).view(1,2),
            'pickup_loc': torch.FloatTensor([(i,j) for i in range(2) for j in range(2)]).view(1,-1,2),
            'dropoff_depot': torch.FloatTensor([3,3]).view(1,2),
            'dropoff_loc': torch.FloatTensor([(i,j) for i in range(3, 5) for j in range(3, 5)]).view(1,-1,2),
            'stack_size': 10,
            'num_stacks': 2
        }
        
        state = StateDTSPMS.initialize(input)
        # print(state)
        
        # Simulate visiting (1,..,N, 0)
        iteration = 0
        node = torch.LongTensor([1])    
        stack = torch.LongTensor([0]) + 2 * state.total_items + 2
    
        for iteration in range(state.total_items):
            state = state.update(node)
            node = node + 1
            state = state.update(stack)
        self.assertTrue(state.finished_pickup())
            
        state = state.update(torch.LongTensor([0]))
        state = state.update(torch.LongTensor([5]))
        self.assertTrue(state.finished_transit())
        
        node = node + state.total_items + 1
        for iteration in range(state.total_items):
            state = state.update(stack)
            node = node - 1
            state = state.update(node)
        
            
        self.assertTrue(state.finished_dropoff())
    
if __name__ == "__main__":
    unittest.main()