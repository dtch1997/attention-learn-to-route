# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:14:48 2020

@author: Daniel Tan
"""


import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter

class StateDTSPPL(NamedTuple):
    # Fixed input
    loc_pickup: torch.Tensor    # Pickup location of object i. Index 0 is the pickup depot
    loc_dropoff: torch.Tensor   # Dropoff location of object i. Index 1 is the dropoff depot
    dist_pickup: torch.Tensor   # Distances between each pickup location; symmetric
    dist_dropoff: torch.Tensor  # Distances between each dropoff location; symmetric

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    items_in_stack: torch.Tensor    # Array of integers, representing items in the stack sfrom bottom (index 0) to top. 
    pickup_visited_: torch.Tensor   # Boolean mask; tracks pickup nodes that have been visited
    dropoff_visited_: torch.Tensor  # Boolean mask; tracks dropoff nodes that have been visited
    lengths: torch.Tensor 
    cur_coord: torch.Tensor
    i: torch.Tensor                 # Keeps track of step
    
    pickup: bool                    # True if we are still in pickup phase, False otherwise
    
    @property
    def pickup_visited(self):
        if self.pickup_visited_.dtype == torch.uint8:
            return self.pickup_visited_
        else:
            return mask_long2bool(self.pickup_visited_, n=self.loc_pickup.size(-2))
        
    @property
    def dropoff_visited(self):
        if self.dropoff_visited_.dtype == torch.uint8:
            return self.dropoff_visited_
        else:
            return mask_long2bool(self.dropoff_visited_, n=self.loc_dropoff.size(-2))
        
    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                items_in_stack=self.items_in_stack[key],
                pickup_visited_=self.pickup_visited_[key],
                dropoff_visited_=self.dropoff_visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
            )
        return super(StateDTSPPL, self).__getitem__(key)
    
    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        pickup_depot = input['pickup_depot']    # Shape (B, 2)
        dropoff_depot = input['dropoff_depot']  # Shape (B, 2)
        pickup_loc = input['pickup_loc']        # Shape (B, N, 2)
        dropoff_loc = input['dropoff_loc']      # Shape (B, N, 2)
        
        # Number of pickup and dropoff locations must be the same
        assert pickup_loc.size(1) == dropoff_loc.size(1)
        batch_size, n_loc, _ = pickup_loc.size()
        
        loc_pickup=torch.cat((pickup_depot[:, None, :], pickup_loc), -2)
        loc_dropoff=torch.cat((dropoff_depot[:, None, :], dropoff_loc), -2)
        
        return StateDTSPPL(
            loc_pickup=loc_pickup,
            dist_pickup=(loc_pickup[:, :, None, :] - loc_pickup[:, None, :, :]).norm(p=2, dim=-1),
            loc_dropoff=lock_dropoff,
            dist_dropoff=(loc_dropoff[:, :, None, :] - loc_dropoff[:, None, :, :]).norm(p=2, dim=-1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            items_in_stack=torch.torch.zeros((batch_size, n_loc), dtype=torch.long, device=pickup_loc.device),
            pickup_visited_= (  
                # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc_pickup.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            dropoff_visited_ = (  
                # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc_pickup.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=pickup_depot[:, None, :],  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device, # Scalar that keeps track of step number
            pickup = True) 
        )
        
    def get_final_cost(self):
        assert self.all_finished()
        # The final cost is the sum of the edge lengths travelled 
        # plus (num_reload) * (cost_per_reload)
        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)


    def update(self, selected, permutation = None):
        """
        :param selected:    torch.Tensor of shape (B,), 
                            Each entry is an index in (0, N-1)
                            
        :param permutation: torch.Tensor of shape (B, k), 
                            Each row is a permutation of (0, 1, ..., k-1) representing how to reload items
        """
        # TODO: Figure out how to implement reloading in this framework
        prev_a = selected[:, None] 
        if not self.finished_pickup():
            loc = self.pickup_loc
            visited = self.pickup_visited_
            # Add the length
            cur_coord = loc[self.ids, selected]
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)
            # Add item to top of stack.
            # Items are indexed from 1 to N, 0 represents lack of item
            # TODO: Add support for reloading
            self.items_in_stack[:, self.i] = selected + 1
            
            # Set the visited mask
            if visited_.dtype == torch.uint8:
                # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
                # Add one dimension since we write a single value
                visited_ = visited_.scatter(-1, prev_a[:, :, None], 1)
            else:
                # This works, will not set anything if prev_a -1 == -1 (depot)
                visited_ = mask_long_scatter(visited_, prev_a - 1)
        
            return self._replace(prev_a=prev_a, pickup_visited_=visited_,
                         lengths=lengths, items_in_stack = items_in_stack, 
                         cur_coord=cur_coord, i=self.i + 1)

        else: # finished pickup
            loc = self.dropoff_loc
            visited = self.dropoff_visited_
            
            


            
        if not self.finished_pickup():
            return self._replace(prev_a=prev_a, pickup_visited_=visited_,
                         lengths=lengths, cur_coord=cur_coord, i=self.i + 1)
        else:
            return self._replace(prev_a=prev_a, dropoff_visited_=visited_,
                         lengths=lengths, cur_coord=cur_coord, i=self.i + 1)

    def finished_pickup(self):
        return (self.items_in_stack >= 0).all()

    def all_finished(self):
        return (self.items_in_stack < 0).all()
    
    def get_current_node(self):
        return self.prev_a
    
    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited 0 = feasible, 1 = infeasible
        :return:
        """
        # Create a mask that is 1 for depot (index 0) and 0 elsewhere
        batch_size, n_loc_include_depot, _ = self.pickup_loc.size()
        depot_mask = torch.zeros((batch_size, n_loc_include_depot), dtype=torch.uint8)
        depot_mask[:,0] += 1
        
        if self.pickup:
            # If we have finished pickup, we must go to the depot. 
            if self.finished_pickup():
                return depot_mask
            # If we are not finished with pickup, only visit univisited nodes
            else:
                if self.pickup_visited_.dtype == torch.uint8:
                    visited_loc = self.pickup_visited_[:, :, 1:]
                else:
                    visited_loc = mask_long2bool(self.pickup_visited_, n=self.demand.size(-1))
                return visited_loc & (1 - depot_mask)
            
        else:
            # If we have finished delivery, we must go to the depot
            if self.all_finished():
                return depot_mask
            # If we are not finished with delivery, we can only visit nodes with the top k items
            else:
                