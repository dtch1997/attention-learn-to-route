# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 00:38:48 2020

@author: Daniel Tan
"""

import torch
import pdb
from problems.dtspms.state_dtspms import StateDTSPMS
from problems.dtspms.problem_dtspms import DTSPMS, DTSPMSDataset

from problems.tsp.problem_tsp import TSP, TSPDataset
from problems.tsp.state_tsp import StateTSP

from problems.dtspss.problem_dtspss import DTSPSS, DTSPSSDataset
from problems.dtspss.problem_dtspss import StateDTSPSS

def test_dtspss():
    input = {
        'pickup_depot': torch.FloatTensor([0, 0]).view(1,2),
        'pickup_loc': torch.FloatTensor([(i,j) for i in range(3) for j in range(3)]).view(1,-1,2),
        'dropoff_depot': torch.FloatTensor([4,4]).view(1,2),
        'dropoff_loc': torch.FloatTensor([(i,j) for i in range(4, 7) for j in range(4, 7)]).view(1,-1,2),
    }
    
    state = StateDTSPSS.initialize(input)
    selected = torch.LongTensor([1]).view(1,)
    for iteration in range(9):
        print(state.get_current_node(), state.lengths, state.get_mask())
        state = state.update(selected)
        selected = selected + 1
    print(state.get_current_node(), state.lengths, state.get_mask())
    state = state.update(torch.LongTensor([0]).view(1,))
    print(state.get_current_node(), state.lengths, state.get_mask())
    state = state.update(torch.LongTensor([10]).view(1,))
    print(state.get_current_node(), state.lengths, state.get_mask())

    # Should be delivering items now
    for iteration in range(9):
        state = state.update(torch.LongTensor([0]).view(1,))
        print(state.get_current_node(), state.lengths, state.get_mask())
    print(state.finished_pickup(), state.finished_dropoff(), state.all_finished())

def test_tsp():
    input = torch.FloatTensor([(i,j) for i in range(3) for j in range(3)]).view(1,-1,2)
    state = StateTSP.initialize(input)
    selected = torch.LongTensor([1]).view(1,)
    for iteration in range(8):
        print(state.get_current_node(), state.lengths, state.get_mask())
        state = state.update(selected)
        selected = selected + 1
    
    
def test_dtspms():
    input = {
        'pickup_depot': torch.FloatTensor([0, 0]).view(1,2),
        'pickup_loc': torch.FloatTensor([(i,j) for i in range(3) for j in range(3)]).view(1,-1,2),
        'dropoff_depot': torch.FloatTensor([4,4]).view(1,2),
        'dropoff_loc': torch.FloatTensor([(i,j) for i in range(4, 7) for j in range(4, 7)]).view(1,-1,2),
        'stack_size': 10,
        'num_stacks': 2
    }
    
    state = StateDTSPMS.initialize(input)
    
    # Simulate visiting (1,..,N, 0)
    iteration = 0
    selected = torch.LongTensor([1]).view(1,)
    stack = torch.LongTensor([0]).view(1,)
    
    #print(state.get_current_node())
    #state = state.update(selected, stack)
    #print(state.get_current_node())
    
    for iteration in range(state.total_items):
        print(state.get_current_node(), state.lengths, state.get_mask())
        state = state.update(selected, stack)
        selected = selected + 1
    print(state.get_current_node(), state.lengths, state.get_mask())
    state = state.update(torch.LongTensor([0]).view(1,), stack)
    print(state.get_current_node(), state.lengths, state.get_mask())
    state = state.update(torch.LongTensor([state.total_items+1]).view(1,), stack)
    print(state.get_current_node(), state.lengths, state.get_mask())
    
    state = state.update(None, stack)
    print(state.get_current_node(), state.lengths, state.get_mask())
    
if __name__ == "__main__":
    test_dtspss()