# -*- coding: utf-8 -*-

import torch
from typing import NamedTuple

class Stack(NamedTuple):   
    """
    A general purpose Stack backed by torch Tensors to facilitate
    deep feature extraction.
        
    Supported methods:
        empty()
        full()
        push()
        peek()
        pop()
        
    Does very little error checking so users should be careful to
    call push() only when the stack is not full, and pop() when the stack
    is not empty. 
    """
    contents: torch.Tensor
    num_stacks: int
    stack_size: int
    
    @staticmethod
    def initialize(size, dtype = torch.int64, device = None):
        return Stack(contents = torch.zeros(size = size, dtype = dtype, device = device),
                     num_stacks = size[-2],
                     stack_size = size[-1])
        
    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                contents = self.contents[key]
            )
        return super(Stack, self).__getitem__(key)
        
    def current_size(self, per_stack = True):
        """
        :return:            torch.Tensor size (B, steps, num_stacks, 1) of 
                            the number of items in every stack
                            
                            OR torch.Tensor size (B, steps) of total items across
                            all stacks if per_stack = False
        """
        if per_stack:
            return (self.contents > 0).sum(dim = -1, keepdim = True)
        else:
            return (self.contents > 0).sum(dim = (-2,-1))
        
    def peek(self, stack_idx = None):
        """
        Peek the top item on the stack(s).
        If empty, return a 0 at the corresponding indices
        
        :param stack_idx:   torch.Tensor size(B, steps, 1) of the stack to pop
                            If none, peek every stack
            
        :return item:       torch.Tensor size(B, steps, num_stacks) of 
                            the top item on the stack(s)
                            
                            or torch.Tensor size(B, steps, 1) if stack_idx set
        """
        last_occupied_idx = self.current_size() - 1
        # If last_occupied_idx = -1, that stack is empty, so return a 0
        last_occupied_idx[last_occupied_idx < 0] = 0
        items = self.contents.gather(-1, last_occupied_idx)
        if stack_idx is not None:
            return items.gather(-2, stack_idx[:,:,:,None])
        else:
            return items
    
    def empty(self, stack_idx = None):
        """
        :return:            torch.Tensor size (B, steps, num_stacks) of booleans
                            True if that stack is empty, False otherwise
        """
        if stack_idx is not None:
            return (self.current_size() == 0).gather(-1, stack_idx)
        return (self.current_size() == 0)
    
    def full(self, stack_idx = None):
        if stack_idx is not None:
            return (self.current_size() == self.stack_size).gather(-1, stack_idx, keepdim=True)
        return self.current_size() == self.stack_size
    
    def push(self, items, stack_idx = None):
        """
        Push items to the stack(s)
        
        :param items:       torch.Tensor size(B, steps, num_stacks, 1) of the item to push
                            or size(B, steps, 1, 1) if stack_idx 
                            Entries have values (1, ..., N)
                            
        :param stack_idx:   torch.Tensor size(B, steps, 1) of the stack to pop
                            If none, push to every stack
            
        :return stack:      Updated stack state after pushing
        """
        #if stack_idx is not None:
        #    assert items.size() == stack_idx.size(), "Number of items must be same as stack index"
        assert self.items_is_valid(items), f"Invalid items {items}"
        assert self.stack_idx_is_valid(stack_idx), f"Invalid stack idx {stack_idx}"
        
        next_empty_idx = self.current_size() 
        # size (B, steps, num_stacks, 1), entries are the next empty stack position
        
        if stack_idx is not None:
            # Have to do a nested scatter here because PyTorch doesn't support 2D scattering

            all_items = self.contents.gather(-1, next_empty_idx)
            assert (all_items == 0).all(), "Index should be empty"
            # Insert items into appropriate locations
            all_items.scatter_(-2, stack_idx[:,:,:,None], items)

        else:
            all_items = items
        # size (B, step, num_stacks, 1)
        
        new_contents = self.contents.detach().clone()
        new_contents.scatter_(-1, next_empty_idx, all_items)
        return self._replace(contents = new_contents)
    
    def pop(self, stack_idx = None):
        """
        Pops an item from the stack(s)
        
        :param stack_idx:   torch.Tensor size(B, steps, 1) of the stack to pop
                            If none, pop every stack
            
        :return stack:      Updated stack state after popping
        :return item:       torch.Tensor size(B, steps, num_stacks) of 
                            the top item on the stack(s)
                            
                            or torch.Tensor size(B, steps, 1) if stack_idx set
        """
        last_occupied_idx = self.current_size() - 1
        # If last_occupied_idx = -1, that stack is empty, so return a 0
        last_occupied_idx[last_occupied_idx < 0] = 0
        
        # In the following code, 'items' is an intermediate variable
        # used only to set the stack contents. 
        # We need this because there is no 2D scattering in PyTorch. 
        items = self.contents.gather(-1, last_occupied_idx)
        # size (B, steps, num_stacks, 1) 
        
        if stack_idx is not None:
            # size (B, steps, num_stacks)
            # Set only the items corresponding to stack_idx to 0
            items.scatter_(-2, stack_idx[:,:,:,None], 0)
        else:
            items.zero_()
        
        new_contents = self.contents.detach().clone()
        new_contents.scatter_(-1, last_occupied_idx, items)
        new_stack = self._replace(
            contents = new_contents    
        )
        return new_stack, self.peek(stack_idx)
    
    def stack_idx_is_valid(self, stack_idx):
        return len(stack_idx.size()) == 3
    
    def items_is_valid(self, items):
        return len(items.size()) == 4
    
if __name__ == "__main__":
    # Test the Stack API
    stack = Stack.initialize((1,1,2,2))
    print(f"Stack_empty: {stack.empty()}")
    print("Pushing item 1 to stack 1...")
    # Push item 1 to stack 1
    items = torch.ones((1,1,1,1), dtype=torch.int64)
    stack_idx = torch.ones((1,1,1), dtype=torch.int64)
    stack = stack.push(items, stack_idx)
    print(f"Stack after pushing item 1 to stack 1: {stack}")
    print(f"Stack_empty: {stack.empty()}")
    
    # Peek the top items
    print(f"Peeking top of stack: {stack.peek()}")
    
    # Pop the top item
    print("Popping stack 1...")
    stack, items = stack.pop(stack_idx)
    print(f"Stack after popping stack 1: {stack}")
    print(f"Popped item: {items}")
        
        
        
        