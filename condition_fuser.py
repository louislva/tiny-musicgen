import einops
from einops import rearrange
from xformers import ops
import torch
from torch import nn
import typing as tp
import torch.nn.functional as F
from audiocraft.modules.streaming import StreamingModule
from transformer import create_sin_embedding

ConditionType = tp.Tuple[torch.Tensor, torch.Tensor]  # condition, mask

class ConditionFuser(StreamingModule):
    """Condition fuser handles the logic to combine the different conditions
    to the actual model input.

    Args:
        fuse2cond (tp.Dict[str, str]): A dictionary that says how to fuse
            each condition. For example:
            {
                "prepend": ["description"],
                "sum": ["genre", "bpm"],
                "cross": ["description"],
            }
        cross_attention_pos_emb (bool, optional): Use positional embeddings in cross attention.
        cross_attention_pos_emb_scale (int): Scale for positional embeddings in cross attention if used.
    """
    FUSING_METHODS = ["sum", "prepend", "cross", "input_interpolate"]

    def __init__(self, fuse2cond: tp.Dict[str, tp.List[str]], cross_attention_pos_emb: bool = False,
                 cross_attention_pos_emb_scale: float = 1.0):
        super().__init__()
        assert all(
            [k in self.FUSING_METHODS for k in fuse2cond.keys()]
        ), f"Got invalid fuse method, allowed methods: {self.FUSING_METHODS}"
        self.cross_attention_pos_emb = cross_attention_pos_emb
        self.cross_attention_pos_emb_scale = cross_attention_pos_emb_scale
        self.fuse2cond: tp.Dict[str, tp.List[str]] = fuse2cond
        self.cond2fuse: tp.Dict[str, str] = {}
        for fuse_method, conditions in fuse2cond.items():
            for condition in conditions:
                self.cond2fuse[condition] = fuse_method

    def forward(
        self,
        input: torch.Tensor,
        conditions: tp.Dict[str, ConditionType]
    ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Fuse the conditions to the provided model input.

        Args:
            input (torch.Tensor): Transformer input.
            conditions (dict[str, ConditionType]): Dict of conditions.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The first tensor is the transformer input
                after the conditions have been fused. The second output tensor is the tensor
                used for cross-attention or None if no cross attention inputs exist.
        """
        B, T, _ = input.shape

        if 'offsets' in self._streaming_state:
            first_step = False
            offsets = self._streaming_state['offsets']
        else:
            first_step = True
            offsets = torch.zeros(input.shape[0], dtype=torch.long, device=input.device)

        assert set(conditions.keys()).issubset(set(self.cond2fuse.keys())), \
            f"given conditions contain unknown attributes for fuser, " \
            f"expected {self.cond2fuse.keys()}, got {conditions.keys()}"
        cross_attention_output = None
        for cond_type, (cond, cond_mask) in conditions.items():
            op = self.cond2fuse[cond_type]
            if op == 'sum':
                input += cond
            elif op == 'input_interpolate':
                cond = einops.rearrange(cond, "b t d -> b d t")
                cond = F.interpolate(cond, size=input.shape[1])
                input += einops.rearrange(cond, "b d t -> b t d")
            elif op == 'prepend':
                if first_step:
                    input = torch.cat([cond, input], dim=1)
            elif op == 'cross':
                if cross_attention_output is not None:
                    cross_attention_output = torch.cat([cross_attention_output, cond], dim=1)
                else:
                    cross_attention_output = cond
            else:
                raise ValueError(f"unknown op ({op})")

        if self.cross_attention_pos_emb and cross_attention_output is not None:
            positions = torch.arange(
                cross_attention_output.shape[1],
                device=cross_attention_output.device
            ).view(1, -1, 1)
            pos_emb = create_sin_embedding(positions, cross_attention_output.shape[-1])
            cross_attention_output = cross_attention_output + self.cross_attention_pos_emb_scale * pos_emb

        if self._is_streaming:
            self._streaming_state['offsets'] = offsets + T

        return input, cross_attention_output
