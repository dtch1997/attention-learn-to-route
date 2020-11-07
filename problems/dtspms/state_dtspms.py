# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:14:48 2020

@author: Daniel Tan
"""


import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
from .stack import Stack

class StateDTSPMS(NamedTuple):
    # Fixed input
    loc: torch.Tensor               # Shape (2n + 2, 2). 
                                    # 0   -> pickup depot
                                    # 1:n -> pickup locations
                                    # n+1 -> dropoff depot
                                    # other -> dropoff locations
    dist_pickup: torch.Tensor       # Distances between each pickup location; symmetric
    dist_dropoff: torch.Tensor      # Distances between each dropoff location; symmetric



    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State variables that will be used in computing the context embedding
    prev_a: torch.Tensor
    visited_: torch.Tensor          # Boolean mask; tracks nodes that have been visited
    lengths: torch.Tensor 
    cur_coord: torch.Tensor
    stack: Stack                    # Keeps track of items on the stack
    next_item_to_unload: torch.Tensor   # The next item to unload, if applicable
    
    # Global variables; these are shared across all batch elements
    i: torch.Tensor                 # Keeps track of step
    num_picked_up: torch.Tensor     # Keeps track of how many items we've picked up
    num_dropped_off: torch.Tensor   # Keeps track of how many items we've dropped off
    pickup: bool
    transit: bool
    dropoff: bool
    next_action_is_stack: bool      # If False, next action is a node action i.e. moving to a new node
                                    # Otherwise, the next action is a stack action i.e. pick a stack to load / unload

    @property
    def total_items(self):
        """Total number of items that must be collected"""
        return (self.loc.size(-2) // 2) - 1
    
    @property
    def batch_size(self):
        return self.loc.size(0)
    
    @property
    def device(self):
        return self.loc.device

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))
        
    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
                stack = self.stack[key],
            )
        return super(StateDTSPMS, self).__getitem__(key)
    
    @staticmethod
    def initialize(input, visited_dtype=torch.uint8, stack_size=20, num_stacks=2):
        pickup_depot = input['pickup_depot']        # Shape (B, 2)
        dropoff_depot = input['dropoff_depot']      # Shape (B, 2)
        pickup_loc = input['pickup_loc']            # Shape (B, N, 2)
        dropoff_loc = input['dropoff_loc']          # Shape (B, N, 2)
        
        # Number of pickup and dropoff locations must be the same
        assert pickup_loc.size(1) == dropoff_loc.size(1), "Unequal number of pickup and dropoff locations"
        batch_size, n_loc, _ = pickup_loc.size()
        # Check that we have enough space in the stacks to pick up all items
        
        # Store depot location as index 0
        loc_pickup=torch.cat((pickup_depot[:, None, :], pickup_loc), -2)
        loc_dropoff=torch.cat((dropoff_depot[:, None, :], dropoff_loc), -2)
        loc = torch.cat((loc_pickup, loc_dropoff), -2)
        
        return StateDTSPMS(
            # Constants
            loc=loc,
            dist_pickup=(loc_pickup[:, :, None, :] - loc_pickup[:, None, :, :]).norm(p=2, dim=-1),
            dist_dropoff=(loc_dropoff[:, :, None, :] - loc_dropoff[:, None, :, :]).norm(p=2, dim=-1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps 
            
            # State variables
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            visited_= (  
                # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, 2*(n_loc + 1),
                    dtype=torch.uint8, device=loc_pickup.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ), 
            lengths=torch.zeros(batch_size, 1, device=loc_pickup.device),
            cur_coord=pickup_depot[:, None, :],  # Add step dimension
            stack = Stack.initialize((batch_size, 1, num_stacks, stack_size), device = loc.device),
            
            # Global variables
            i=torch.zeros(1, dtype=torch.int64, device=loc_pickup.device), # Scalar that keeps track of step number
            num_picked_up = torch.zeros(1, dtype = torch.int64, device = loc_pickup.device),
            num_dropped_off = torch.zeros(1, dtype = torch.int64, device = loc_pickup.device),
            pickup = True,  
            transit = False,
            dropoff = False,
            next_action_is_stack = False,    # On the first action, we need to select a node to move to. 
            next_item_to_unload = torch.zeros((batch_size, 1), dtype=torch.int64, device = loc.device)
        )
        
    def get_final_cost(self):
        # This function should be called once the agent has returned to dropoff depot at the end. 
        assert self.all_finished()
        # The final cost is the sum of the edge lengths travelled 
        return self.lengths
    
    #------------------------#
    # State update functions #
    #------------------------#

    def update(self, selected):
        """
        This API is kept the same as the other routing problems. 
        """
        return StateDTSPMS._update(self, selected)
            
    @staticmethod
    def _update(state: 'StateDTSPMS', action):
        """
        Simulates a step in the DTSPMS. 
        A step can consist of either a node action (indices 0:2n+2) or a stack action (indices 2n+2:2n+4)
        There is almost no error checking done; we trust that the selected action is valid for the current state. 
        
        Adhering to a functional style (no mutable state) helps prevent bugs. 
        
        :param action:      torch.Tensor of shape (B,), 
                            Entries in (0, 1, ..., 2n+3) representing which node to travel to next
        
        :return:            new StateDTSPMS after completing the action
        """
        if state.pickup: 
            if state.next_action_is_stack: 
                # pickup phase, stack action
                stack_idx = action - (2 * state.total_items + 2)
                # shape (B,) -> (B, 1, 1)
                stack_idx = stack_idx[:,None,None]
                
                item = state.get_current_node()
                # shape (B, 1) -> (B, 1, 1, 1)
                item = item[:,None,None]
                
                # Load item, set the next action to be a node action
                state = StateDTSPMS._load(state, item, stack_idx)
                state = StateDTSPMS._set_next_action(state, stack = False)
            else: 
                # pickup phase, node action
                new_node = action
                
                # Move to new node, set the next action to be a stack action
                state = StateDTSPMS._move(state, new_node)
                state = StateDTSPMS._set_next_action(state, stack = True)
                
            if state.finished_pickup():
                # We have just completed the final pickup
                # Enter the transit phase
                state = StateDTSPMS._set_pickup(state, False)
                state = StateDTSPMS._set_transit(state, True)
                
            return state
        
        elif state.transit:
            # transit phase.
            # Move from final pickup
            assert not (state.next_action_is_stack)
            new_node = action
            state = StateDTSPMS._move(state, new_node)
            
            if state.finished_transit():
                state = StateDTSPMS._set_transit(state, False)
                state = StateDTSPMS._set_dropoff(state, True)
                # In the delivery phase, we must take n stack actions
                state = StateDTSPMS._set_next_action(state, stack = True)
            return state
                
        elif state.dropoff:
            # dropoff phase.
            
            if state.next_action_is_stack:
                # Select a stack to unload
                stack_idx = action - (2 * state.total_items + 2)
                stack_idx = stack_idx[:,None,None]
                state, item = StateDTSPMS._unload(state, stack_idx)
                state = StateDTSPMS._set_next_action(state, stack = False)
                state = StateDTSPMS._set_next_item(state, item)
                
            else:
                # Unload previously selected stack
                next_item = state.next_item_to_unload              
                dest_node = next_item[:,0,0,0] + state.total_items + 1
                new_node = action
                assert (new_node == dest_node).all(), f"{new_node}, {dest_node}"
                state = StateDTSPMS._move(state, new_node)
                state = StateDTSPMS._set_next_action(state, stack = True)
                state = StateDTSPMS._set_next_item(state, torch.zeros_like(next_item) -1)
        
            if state.finished_dropoff():
                # After completing deliveries, we must take a single node action
                # to return to the supply depot.
                state = StateDTSPMS._set_dropoff(state, False)
                state = StateDTSPMS._set_next_action(state, stack = False)
            return state
        else: 
            # return phase
            # Agent must return to the supply depot. 
            assert not state.next_action_is_stack
            new_node = action
            state = StateDTSPMS._move(state, new_node)
            assert state.all_finished()
            return state
    
    @staticmethod
    def distance(loc1, loc2):
        """
        Euclidean distance between two coordinates
        loc1, loc2: torch.Tensor of size (...., coord_dim)
        
        Return: torch.Tensor of size (.....,) of Euclidean distance
        """
        return (loc1 - loc2).norm(p=2, dim=-1)
    
    @staticmethod
    def _move(state, new_node):
        new_node = new_node[:, None]
        prev_coord = state.cur_coord
        cur_coord = state.loc[state.ids, new_node]
        lengths = state.lengths + StateDTSPMS.distance(cur_coord, prev_coord)
        
        # Update the visited mask
        visited_ = state.visited_
        if visited_.dtype == torch.uint8:
            # Add one dimension since we write a single value
            visited_ = visited_.scatter(-1, new_node[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(visited_, new_node)  
            
        return state._replace(
            prev_a = new_node,
            cur_coord = cur_coord, 
            lengths = lengths,
            visited_ = visited_
        )

    @staticmethod
    def _load(state, item, stack_idx):
        new_stack = state.stack.push(item, stack_idx)
        return state._replace(
            stack = new_stack,
            num_picked_up = state.num_picked_up + 1
        )
    
    @staticmethod
    def _unload(state, stack_idx):
        new_stack, item = state.stack.pop(stack_idx)
        return state._replace(
            stack = new_stack,
            num_dropped_off = state.num_dropped_off + 1
        ), item
    
    @staticmethod
    def _set_next_action(state, stack: bool):
        return state._replace(
            next_action_is_stack = stack
        )
    
    @staticmethod
    def _set_next_item(state, item):
        return state._replace(
            next_item_to_unload = item    
        )
    
    @staticmethod
    def _set_pickup(state, pickup: bool):
        return state._replace(
            pickup = pickup    
        )
    
    @staticmethod
    def _set_transit(state, transit: bool):
        return state._replace(
            transit = transit
        )
    
    @staticmethod
    def _set_dropoff(state, dropoff: bool):
        return state._replace(
            dropoff = dropoff  
        )
    
    
    #------------------------#
    # State update functions #
    #------------------------#

    def finished_pickup(self):
        return self.num_picked_up.item() >= self.total_items
    
    def finished_transit(self):
        return self.finished_pickup() \
            and (self.get_current_node() >= self.total_items + 1).all()
    
    def finished_dropoff(self):
        return self.num_dropped_off.item() >= self.total_items \
            and (self.next_item_to_unload < 0).all()

    def all_finished(self):
        return self.finished_pickup() and self.finished_dropoff() \
            and (self.prev_a == self.total_items + 1).all()
    
    def get_current_node(self):
        return self.prev_a
    
    def get_node_mask(self):
        """
        0 for feasible, 1 for infeasible 
        """
        
        pickup_depot_mask = torch.ones_like(self.visited)
        pickup_depot_mask[:,:,0] = 0
        delivery_depot_mask = torch.ones_like(self.visited)
        delivery_depot_mask[:,:,self.total_items + 1] = 0
        
        pickup_loc_mask = (1-pickup_depot_mask).detach().clone()
        pickup_loc_mask[:,:,self.total_items + 1:] = 1
        
        mask = None
        if self.pickup:
            mask = pickup_loc_mask | self.visited
        elif self.transit:
            if (self.get_current_node() == 0).all():
                # We are at the pickup depot
                # Can only move to delivery depot
                mask = delivery_depot_mask
            else:
                # We need to move to the pickup depot
                mask = pickup_depot_mask
        elif self.dropoff:
            next_item = self.next_item_to_unload[:,:,0,0]
            # assert (next_item > 0).all()
            new_node = next_item + self.total_items + 1
            mask = torch.ones_like(self.visited)
            mask.scatter_(-1, new_node[:,:,None], 0)
            
        else:
            mask = delivery_depot_mask
            
        # Extend to stack actions; which are infeasible
        return torch.cat(
            (mask, torch.ones_like(mask[:,:,:2])), 
            dim = -1
        )

    def get_stack_mask(self):
        if self.pickup:
            # Mask should be 0 for any stack with remaining space
            mask = self.stack.full()
        else:
            # Mask should be 0 for any stack with remaining items
            mask = self.stack.empty()
        mask = mask[:,:,:,0].to(torch.uint8)
        return torch.cat(
            (torch.ones_like(self.visited), mask),
            dim = -1
        )        

    def get_mask(self):
        if self.next_action_is_stack:
            return self.get_stack_mask()
        else:
            return self.get_node_mask()
    
