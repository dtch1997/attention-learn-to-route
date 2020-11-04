# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:14:48 2020

@author: Daniel Tan
"""


import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter

class StateDTSPMS(NamedTuple):
    # Fixed input
    loc_pickup: torch.Tensor    # Pickup location of object i. Index 0 is the pickup depot
    loc_dropoff: torch.Tensor   # Dropoff location of object i. Index 1 is the dropoff depot
    dist_pickup: torch.Tensor   # Distances between each pickup location; symmetric
    dist_dropoff: torch.Tensor  # Distances between each dropoff location; symmetric
    pickup: bool
    num_stacks: int
    stack_size: int

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State; each of these will have a shape of (B,...) - different value per batch element
    prev_a: torch.Tensor
    items_in_stack: torch.Tensor    # Array of integers, representing items in the stack sfrom bottom (index 0) to top. 
    pickup_visited_: torch.Tensor   # Boolean mask; tracks pickup nodes that have been visited
    dropoff_visited_: torch.Tensor  # Boolean mask; tracks dropoff nodes that have been visited
    lengths: torch.Tensor 
    cur_coord: torch.Tensor
    
    # Global variables; these are shared across all batch elements
    i: torch.Tensor                 # Keeps track of step
    num_picked_up: torch.Tensor     # Keeps track of how many items we've picked up
    num_dropped_off: torch.Tensor   # Keeps track of how many items we've dropped off
    
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
        
    @property
    def num_items_by_stack(self):
        """
        :return:    torch.Tensor (B, num_stacks), integer of how many items in each stack
        """
        return (self.items_in_stack > 0).sum(dim=2)
    
    @property
    def num_items(self):
        """
        :return:    torch.Tensor (B,), integer of how many items in total
        """
        return self.num_items_by_stack.sum(dim=1)
        
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
        stack_size = input['stack_size']
        num_stacks = input['num_stacks']
        
        # Number of pickup and dropoff locations must be the same
        assert pickup_loc.size(1) == dropoff_loc.size(1), "Unequal number of pickup and dropoff locations"
        batch_size, n_loc, _ = pickup_loc.size()
        # Check that we have enough space in the stacks to pick up all items
        assert num_stacks * stack_size >= n_loc, "Total item capacity is insufficient to pick up all items"
        # Store depot location as index 0
        loc_pickup=torch.cat((pickup_depot[:, None, :], pickup_loc), -2)
        loc_dropoff=torch.cat((dropoff_depot[:, None, :], dropoff_loc), -2)
        
        return StateDTSPMS(
            # Constants
            loc_pickup=loc_pickup,
            dist_pickup=(loc_pickup[:, :, None, :] - loc_pickup[:, None, :, :]).norm(p=2, dim=-1),
            loc_dropoff=lock_dropoff,
            dist_dropoff=(loc_dropoff[:, :, None, :] - loc_dropoff[:, None, :, :]).norm(p=2, dim=-1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps 
            stack_size = stack_size,
            num_stacks = num_stacks,
            
            # State variables
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            items_in_stack=torch.torch.zeros((batch_size, num_stacks, stack_size), dtype=torch.long, device=pickup_loc.device),
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
            lengths=torch.zeros(batch_size, 1, device=loc_pickup.device),
            cur_coord=pickup_depot[:, None, :],  # Add step dimension
            
            # Global variables
            i=torch.zeros(1, dtype=torch.int64, device=loc_pickup.device), # Scalar that keeps track of step number
            num_picked_up = torch.zeros(1, dtype = torch.int64, device = loc_pickup.device),
            num_dropped_off = torch.zeros(1, dtype = torch.int64, device = loc_pickup.device),
            pickup = True
        )
        
    def get_final_cost(self):
        assert self.all_finished()
        # The final cost is the sum of the edge lengths travelled 
        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)


    def update(self, selected, stack):
        """
        Simulates a step in the DTSPMS. 
        
        During pickup, a step consists of moving to a valid node, picking up an item, and selecting a stack to place it on.
        During delivery, a step consists of selecting a stack, removing the top item, and delivering it to that node.  
        
        :param selected:    torch.Tensor of shape (B,), 
                            Entries in (0, 1, ..., N-1) representing which node to travel to next
                            
        :param stack:       torch.Tensor of shape (B,), 
                            Entries in (0, 1, ..., num_stacks-1) representing which stack to put next item on for pickup
        """

        if self.pickup:
            pickup = True
            prev_a = selected[:, None] 
            loc = self.pickup_loc
            visited = self.pickup_visited_
            
            # Move to new position, add the length
            cur_coord = loc[self.ids, selected] # Shape (B, 2)
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)
            
            if self.finished_pickup():
                assert (selected == 0).all(), "Must move to depot after finishing pickup"
                assert (cur_coord == self.pickup_loc[:, 0]).all(), "Must be at depot"
                # We are now back at the pickup depot
                # Teleport to the dropoff depot
                cur_coord = self.dropoff_loc[:, 0, :]
                prev_a = selected[:, None] + self.pickup_node.
                # In the next update, we set pickup to be False
                pickup = False
            
            
            # Add item to top of stack.
            # TODO: Add error checking for whether stack is full. 
            # First, get the first empty position in each stack
            stack_next_empty_idx = self.num_items_by_stack    
            # shape (B, num_stacks) and entries in (0, .., stack_size-1)
            # Numpy and PyTorch expect a tuple of coordinate arrays for multidimensional indexing
            stack_idx = stack_next_empty_idx[self.ids, stack] 
            # shape (B,) and entries in (0, ..., stack_size-1)
            # Items are indexed from 1 to N, 0 represents lack of item
            items_in_stack = self.items_in_stack.detach().clone()
            items_in_stack[self.ids, stack, stack_idx] = selected + 1
            
            # Set the visited mask
            if visited_.dtype == torch.uint8:
                # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
                # Add one dimension since we write a single value
                visited_ = visited_.scatter(-1, prev_a[:, :, None], 1)
            else:
                # This works, will not set anything if prev_a -1 == -1 (depot)
                visited_ = mask_long_scatter(visited_, prev_a - 1)
                
            # Increment global variables
            i = self.i + 1
            num_picked_up = self.num_picked_up + 1
        
            return self._replace(prev_a=prev_a, pickup_visited_=visited_,
                         lengths=lengths, items_in_stack = items_in_stack, 
                         cur_coord=cur_coord, i=i, num_picked_up = num_picked_up, 
                         pickup = pickup)

        else: 
            # dropoff
            loc = self.dropoff_loc
            visited = self.dropoff_visited_

            stack_last_occupied_idx = self.num_items_by_stack - 1
            # Numpy and PyTorch expect a tuple of coordinate arrays for multidimensional indexing
            stack_idx = stack_last_occupied_idx[self.ids, stack] 
            item_to_deliver = self.items_in_stack[self.ids, stack, stack_idx]
            
            # Ignore argument selected; next location is determined by item removed
            selected = item_to_deliver - 1
            prev_a = selected[:, None] 
            
            # Move to new position, add the length
            cur_coord = loc[self.ids, selected]
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)            
                        
            # Remove items from stack
            # TODO: Add error checking for whether stack is empty
            items_in_stack = self.items_in_stack.detach().clone()
            items_in_stack[self.ids, stack, stack_idx] = 0
            
            # Increment global variables
            i = self.i + 1
            num_dropped_off = self.num_dropped_off + 1
            
            if self.finished_dropoff():
                # We have just delivered the last item, teleport to the depot
                cur_coord = self.dropoff_loc[;, 0, :]
            
            return self._replace(prev_a=prev_a, dropoff_visited_=visited_,
                         lengths=lengths, items_in_stack = items_in_stack, 
                         cur_coord=cur_coord, i=self.i + 1)
            
            


            
        if not self.finished_pickup():
            return self._replace(prev_a=prev_a, pickup_visited_=visited_,
                         lengths=lengths, cur_coord=cur_coord, i=self.i + 1)
        else:
            return self._replace(prev_a=prev_a, dropoff_visited_=visited_,
                         lengths=lengths, cur_coord=cur_coord, i=self.i + 1)

    def finished_pickup(self):
        total_items = self.loc_pickup.size(1) - 1
        return self.num_picked_up.item() >= total_items
    
    def finished_dropoff(self):
        total_items = self.loc_pickup.size(1) - 1
        return self.num_dropped_off.item() >= total_items

    def all_finished(self):
        return self.finished_pickup() and self.finished_dropoff() \
            and (self.cur_coord == self.dropoff_loc[:, 0, :]).all()
    
    def get_current_node(self):
        return self.prev_a
    
    def get_mask(self):
        """
        Gets boolean masks for feasible movements and stacks
        
        :return:    node_mask, stack_mask 
                    node_mask is a (batch_size, n_loc + 1) mask of valid nodes to move to
                    stack_mask is a (batch_size, num_stacks) mask of valid stacks to put items on
        """
        # Create a mask that is 1 for depot (index 0) and 0 elsewhere
        batch_size, n_loc_include_depot, _ = self.pickup_loc.size()
        depot_mask = torch.zeros((batch_size, n_loc_include_depot), dtype=torch.uint8)
        depot_mask[:,0] += 1
        
        total_items = self.loc_pickup.size(1) - 1
        stack_mask = self.num_items_by_stack < total_items
        # shape (B, num_stacks)
        
        node_mask = None
        
        if self.pickup:
            # If we have just finished pickup, we must go to the depot. 
            # pickup will be set to False in the next update step. 
            if self.finished_pickup():
                node_mask = depot_mask
            # If we are not finished with pickup, only visit univisited nodes
            else:
                if self.pickup_visited_.dtype == torch.uint8:
                    visited_loc = self.pickup_visited_[:, :, 1:]
                else:
                    visited_loc = mask_long2bool(self.pickup_visited_, n=self.demand.size(-1))
                node_mask = visited_loc & (1 - depot_mask)
            
        else:
            # If we have finished delivery, we must go to the depot
            if self.finished_dropoff():
                return depot_mask
            # If we are not finished with delivery, we can only visit nodes with the top k items
            else:
                batch_size = self.loc_pickup.size(0)
                stack_last_occupied_idx = self.num_items_by_stack - 1
                # shape (B, num_stacks), entries in (0, ..., stack_size-1)
                # 

                
                
                
                
                # Numpy and PyTorch expect a tuple of coordinate arrays for multidimensional indexing
                item_to_deliver = self.items_in_stack[self.ids, stack, stack_idx]
                
                