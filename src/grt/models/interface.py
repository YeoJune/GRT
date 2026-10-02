from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class ModelOutput:
    logits: Tensor

def validate_inputs(input_ids, attention_mask, segment_len, vocab_size):
    if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0 or input_ids.shape[1] % segment_len:
        raise ValueError("input_ids must be nonempty [B,L], with L divisible by segment_len")
    if input_ids.dtype != torch.int64 or (input_ids < 0).any() or (input_ids >= vocab_size).any():
        raise ValueError("input_ids must be int64 token IDs within the vocabulary")
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape or attention_mask.dtype != torch.bool or attention_mask.device != input_ids.device:
            raise ValueError("attention_mask must be bool [B,L] on the input device")
        if not attention_mask.all():
            raise ValueError("Padding is not supported; attention_mask must be all true")
