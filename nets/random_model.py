import torch
from torch import nn
import math
from utils.tensor_functions import compute_in_batches

from torch.nn import DataParallel
from utils.functions import sample_many


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)

class RandomModel(nn.Module):

    def __init__(self,
                 problem,
                 shrink_size = None):
        super(RandomModel, self).__init__()
        self.problem = problem
        self.temp = 1.0
        self.decode_type = None
        self.shrink_size = shrink_size
        
    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        _log_p, pi = self._inner(input)

        cost, mask = self.problem.get_costs(input, pi)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return cost, ll, pi

        return cost, ll

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) / ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _inner(self, input):

        outputs = []
        sequences = []

        state = self.problem.make_state(input)
        batch_size = state.ids.size(0)

        # Perform decoding steps
        i = 0
        while not (self.shrink_size is None and state.all_finished()):

            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                # (otherwise batch norm will not work well and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Filter states
                    state = state[unfinished]

            log_p, mask = self._get_log_p(state)
            
            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
            state = state.update(selected)

            # Now make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)
    
    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        return sample_many(
            lambda input: self._inner(input),  
            lambda input, pi: self.problem.get_costs(input, pi),  # Don't need embeddings as input to get_costs
            input,
            batch_rep, iter_rep
        )
    
    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _get_log_p_topk(self, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, state, normalize=True):
        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)
        log_p = torch.ones_like(mask).to(torch.float32)
        log_p[mask.to(torch.bool)] = -math.inf

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask


