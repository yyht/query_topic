
import torch
import io
import torch.nn.functional as F
import random
import numpy as np
import time
import math
import datetime
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import logging
import warnings

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None


class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training
    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module
        Args:
            x (`torch.tensor`): The input tensor to apply dropout
        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob

class RobertaClassifier(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, dropout_prob, num_labels, dropout_type="stable"):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.num_labels = num_labels
        
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        if dropout_type == 'stable':
            self.dropout = StableDropout(self.dropout_prob)
            logger.info("++RobertaClassifier++ apply stable dropout++")
        else:
            self.dropout = nn.Dropout(self.dropout_prob)
            logger.info("++RobertaClassifier++ apply normal dropout++")
        
        self.out_proj = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, features, **kwargs):
        if len(features.shape) == 3:
            x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        else:
            x = features
        x = self.dropout(x)
        x = self.dense(x)
        last_rep = torch.tanh(x)
        last_rep = self.dropout(last_rep)
        logits = self.out_proj(last_rep)
        return logits

#------------------------------
#   The Transformer
#------------------------------
class MyBaseModel(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        
        self.transformer = model
        self.config = config
        
    def forward(self, input_ids, input_mask, segment_ids=None, return_mode='cls'):
        model_outputs = self.transformer(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
        hidden_states = model_outputs[0]
        if return_mode == 'cls':
            return hidden_states[:, 0, :]
        elif return_mode == 'mean_pooling':
            input_mask = input_mask.float().unsqueeze(dim=1) # [batch_size, seq_len, 1]
            return torch.sum(input_mask*hidden_states, axis=1) / (1e-10+torch.sum(input_mask, axis=1))
       
class HYM(nn.Module):
    def __init__(self, net,
                 n_classes, contrast_k, contrast_t, device):
        super().__init__()

        self.net = net
        
        self.K = contrast_k
        self.T = contrast_t
        self.dim = n_classes

        # create the queue
        init_logit = torch.randn(self.dim, self.K).to(device)
        self.register_buffer("queue_logit", init_logit)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long).to(device))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, logits):
        # gather logits before updating queue
        batch_size = logits.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the logits at ptr (dequeue and enqueue)
        self.queue_logit[:, ptr : ptr + batch_size] = logits.T

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
        
    def forward(self, input_ids, input_mask, device,
                segment_ids=None,
                dist=None,
                transformer_mode='cls', 
                return_mode='joint',
                normalization=True,
                evaluation=False):
        ce_logit, hidden_states = self.net(
                input_ids=input_ids, 
                input_mask=input_mask, 
                segment_ids=segment_ids, 
                transformer_mode=transformer_mode)
        
        if return_mode == 'joint':
            if normalization:
                prob = nn.functional.normalize(ce_logit, dim=1)
            else:
                prob = ce_logit
            # positive logits: Nx1
            l_pos = dist * prob  # NxC
            l_pos = torch.logsumexp(l_pos, dim=1, keepdim=True)  # Nx1
            # negative logits: NxK
            if normalization:
                normalized_buffer = nn.functional.normalize(self.queue_logit.clone().detach(), dim=0)
            else:
                normalized_buffer = self.queue_logit.clone().detach()
            l_neg = torch.einsum("nc,ck->nck", [dist, normalized_buffer])  # NxCxK
            l_neg = torch.logsumexp(l_neg, dim=1)  # NxK

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

            # dequeue and enqueue
            if not evaluation:
                self._dequeue_and_enqueue(ce_logit)

            return ce_logit, hidden_states, logits, labels
        
        elif return_mode == "joint_flatclr":
            if normalization:
                prob = nn.functional.normalize(ce_logit, dim=1)
            else:
                prob = ce_logit
            # positive logits: Nx1
            l_pos = dist * prob  # NxC
            l_pos = torch.logsumexp(l_pos, dim=1, keepdim=True)  # Nx1
            # negative logits: NxK
            if normalization:
                normalized_buffer = nn.functional.normalize(self.queue_logit.clone().detach(), dim=0)
            else:
                normalized_buffer = self.queue_logit.clone().detach()
            l_neg = torch.einsum("nc,ck->nck", [dist, buffer])  # NxCxK
            l_neg = torch.logsumexp(l_neg, dim=1)  # NxK

            # https://github.com/Junya-Chen/FlatCLR/blob/main/flatclr.py
            logits = (l_neg - l_pos)/self.T # (N, K)

            # labels: positive key indicators
            labels = torch.zeros(l_pos.shape[0], dtype=torch.long).to(device)

            # dequeue and enqueue
            if not evaluation:
                self._dequeue_and_enqueue(ce_logit)
                
            return ce_logit, hidden_states, logits, labels
            
            
            