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
    loc: torch.Tensor               # Shape (2N + 2, 2). 
                                    # 0   -> pickup depot
                                    # 1-N -> pickup locations
                                    # N+1 -> dropoff depot
                                    # other -> dropoff locations
    dist_pickup: torch.Tensor       # Distances between each pickup location; symmetric
    dist_dropoff: torch.Tensor      # Distances between each dropoff location; symmetric
    num_stacks: int
    stack_size: int

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State; each of these will have a shape of (B,...) - different value per batch element
    prev_a: torch.Tensor
    items_in_stack: torch.Tensor    # Array of integers, representing items in the stack sfrom bottom (index 0) to top. 
    visited_: torch.Tensor          # Boolean mask; tracks nodes that have been visited
    lengths: torch.Tensor 
    cur_coord: torch.Tensor
    
    # Global variables; these are shared across all batch elements
    i: torch.Tensor                 # Keeps track of step
    num_picked_up: torch.Tensor     # Keeps track of how many items we've picked up
    num_dropped_off: torch.Tensor   # Keeps track of how many items we've dropped off
    pickup: bool                    # True if we are still in pickup phase, False otherwise
    
    
    @property
    def total_items(self):
        return (self.loc.size(-2) // 2) - 1
    @property
    def total_locations(self):
        return self.loc.size(-2)    
    @property
    def batch_size(self):
        return self.loc.size(0)
    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))
    @property
    def loc_dropoff_depot(self):
        return self.loc[:,self.total_items+1,:]
        
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
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
                pickup=self.pickup,
            )
        return super(StateDTSPMS, self).__getitem__(key)
    
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
        loc = torch.cat((loc_pickup, loc_dropoff), -2)
        
        return StateDTSPMS(
            # Constants
            loc=loc,
            dist_pickup=(loc_pickup[:, :, None, :] - loc_pickup[:, None, :, :]).norm(p=2, dim=-1),
            dist_dropoff=(loc_dropoff[:, :, None, :] - loc_dropoff[:, None, :, :]).norm(p=2, dim=-1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps 
            stack_size = stack_size,
            num_stacks = num_stacks,
            
            # State variables
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            items_in_stack=torch.torch.zeros((batch_size, num_stacks, stack_size), dtype=torch.long, device=pickup_loc.device),
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
            
            # Global variables
            i=torch.zeros(1, dtype=torch.int64, device=loc_pickup.device), # Scalar that keeps track of step number
            num_picked_up = torch.zeros(1, dtype = torch.int64, device = loc_pickup.device),
            num_dropped_off = torch.zeros(1, dtype = torch.int64, device = loc_pickup.device),
            pickup = True
        )
        
    def get_final_cost(self):
        # This function should be called once the agent has returned to dropoff depot at the end. 
        assert self.all_finished()
        # The final cost is the sum of the edge lengths travelled 
        return self.lengths

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
            # Make sure we selected a pickup index
            assert (selected < self.total_items + 1).all(), "Only pickup locations can be selected during pickup"
            pickup = True
            prev_a = selected[:, None] 
            
            # Move to new position, add the length
            cur_coord = self.loc[self.ids, selected] # Shape (B, 2)
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)
            
            if self.finished_pickup():
                assert (selected == 0).all(), "Must move to depot after finishing pickup"
                assert (cur_coord == self.pickup_loc[:, 0]).all(), "Must be at depot"
                # We are now back at the pickup depot
                # Teleport to the dropoff depot
                # Movement from pickup to dropoff depot is not counted in the trip
                cur_coord = self.dropoff_loc[:, 0, :]
                prev_a = selected[:, None]
                # Transition to the dropoff phase
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
            if self.visited_.dtype == torch.uint8:
                # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
                # Add one dimension since we write a single value
                visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
            else:
                # This works, will not set anything if prev_a -1 == -1 (depot)
                visited_ = mask_long_scatter(self.visited_, prev_a - 1)
                
            # Increment global variables
            i = self.i + 1
            num_picked_up = self.num_picked_up + 1
        
            return self._replace(prev_a=prev_a, visited_=visited_,
                         lengths=lengths, items_in_stack = items_in_stack, 
                         cur_coord=cur_coord, i=i, num_picked_up = num_picked_up, 
                         pickup = pickup)

        else: 
            # dropoff
            assert (selected >= self.total_items + 1).all(), \
                "Only dropoff locations can be selected during dropoff"
            stack_last_occupied_idx = self.num_items_by_stack - 1
            # Numpy and PyTorch expect a tuple of coordinate arrays for multidimensional indexing
            stack_idx = stack_last_occupied_idx[self.ids, stack] 
            item_to_deliver = self.items_in_stack[self.ids, stack, stack_idx]
            
            # Ignore argument selected; next location is determined by item removed
            selected = item_to_deliver - 1
            prev_a = selected[:, None] 
            
            # Move to new position, add the length
            cur_coord = self.loc[self.ids, selected]
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)            
                        
            # Remove items from stack
            # TODO: Add error checking for whether stack is empty
            items_in_stack = self.items_in_stack.detach().clone()
            items_in_stack[self.ids, stack, stack_idx] = 0
            
            # Set the visited mask
            if self.visited_.dtype == torch.uint8:
                # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
                # Add one dimension since we write a single value
                visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
            else:
                # This works, will not set anything if prev_a -1 == -1 (depot)
                visited_ = mask_long_scatter(self.visited_, prev_a - 1)
            
            # Increment global variables
            i = self.i + 1
            num_dropped_off = self.num_dropped_off + 1
            
            if self.finished_dropoff():
                # On dropping off the last item, return to depot. 
                # This case is handled separately because after dropping off last item,
                # all stacks are empty so the stack mask is infeasible for all
                selected = torch.ones_like(self.idx) * (self.total_items + 1)
                # shape (B,), entries are all N+1
                prev_a = selected[:, None]
                lengths = lengths + (self.dropoff_depot - cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)            
                cur_coord = self.loc[self.ids, selected]
            
            return self._replace(prev_a=prev_a, visited_=visited_,
                         lengths=lengths, items_in_stack = items_in_stack, 
                         cur_coord=cur_coord, i=self.i + 1, 
                         num_dropped_off = num_dropped_off)

    def finished_pickup(self):
        total_items = self.loc_pickup.size(1) - 1
        return self.num_picked_up.item() >= total_items
    
    def finished_dropoff(self):
        total_items = self.loc_pickup.size(1) - 1
        return self.num_dropped_off.item() >= total_items

    def all_finished(self):
        return self.finished_pickup() and self.finished_dropoff() \
            and (self.cur_coord == self.loc_dropoff_depot).all()
    
    def get_current_node(self):
        return self.prev_a
    
    def get_node_mask(self):
        """
        Gets a (B, 2N+2) mask with the feasible actions (0 = depot).
        0 = feasible, 1 = infeasible
        :return:
        """
        
        # A mask that makes all pickup locations feasible and dropoff locations infeasible
        pickup_mask = torch.arange(2).repeat_interleave(self.total_items + 1) \
                                     .view(1,-1) \
                                     .repeat(self.batch_size)
        # A mask that makes only the pickup depot feasible
        depot_mask = torch.ones((self.batch_size, self.total_items+1), dtype=torch.uint8)
        depot_mask[:,0] -= 1
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))

        if self.pickup:
            depot_mask = torch.cat(depot_mask, torch.ones_like(depot_mask), dim=1)
            if self.finished_pickup(): 
                # Can only go to depot
                return depot_mask
            else:
                # Cannot go to depot, visited nodes, or dropoff nodes
                return (1 - depot_mask) | visited_loc | pickup_mask
        
        else:
            # For dropoff we ignore the node selection anyway
            # So here we return all zeros -> all nodes feasible
            return torch.zeros((self.batch_size, 2 * (self.total_items + 1)), dtype=torch.uint8)
        
    def get_stack_mask(self):
        """
        Gets a (B, stack_size) mask with feasible stacks. 0 = feasible, 1 = infeasible
        
        For pickup: 
            feasible stacks are non-full stacks
        For dropoff:
            feasible stacks are non-empty stacks
        """
        
        if self.pickup:
            # feasible stacks are non-full stacks
            # set 1 for all full stacks
            return self.num_items_by_stack() == self.stack_size
        else:
            # feasible stacks are non-empty stacks
            # set 1 for all empty stacks
            return self.num_items_by_stack == 0
    
    def get_mask(self):
        """
        Gets boolean masks for feasible movements and stacks
        
        :return:    node_mask, stack_mask 
                    node_mask is a (batch_size, n_loc + 1) mask of valid nodes to move to
                    stack_mask is a (batch_size, num_stacks) mask of valid stacks to put items on
        """
        return self.get_node_mask(), self.get_stack_mask()
                
                