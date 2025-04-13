import torch.nn as nn
from .attention import FactorizedSpatioTemporalBlock, LayerNorm

class VideoDecoder(nn.Module):
    """
    A causal decoder composed of multiple FactorizedSpatioTemporalBlocks for autoregressive prediction.

    This module processes decoder tokens `S` using causal self-attention followed by
    cross-attention to encoder outputs. It supports spatial and temporal relative
    positional encoding, and optional position encoding via PEG.

    Args:
        dim (int): Dimensionality of input and output token embeddings.
        depth (int): Number of transformer blocks.
        heads (int): Number of attention heads per block.
        dim_head (int): Dimension of each attention head.
        ff_mult (int): Expansion factor in the feedforward network.
        attn_dropout (float): Dropout applied to attention weights.
        ff_dropout (float): Dropout applied in feedforward layers.
        use_rel_pos_spatial (bool): Whether to apply 2D relative positional bias in spatial attention.
        use_rel_pos_temporal (bool): Whether to apply 1D relative positional bias in temporal attention.
        use_peg_spatial_layers (List[int], optional): Indices of blocks to apply spatial PEG.
        use_peg_temporal_layers (List[int], optional): Indices of blocks to apply temporal PEG.
        dim_context (int, optional): Dimensionality of the encoder output used for cross-attention.
        attn_num_null_kv (int): Number of learnable null key/value pairs to add during cross-attention.
        use_cross_attn_spatial (bool): Whether to enable spatial cross-attention in all blocks.
        use_cross_attn_temporal (bool): Whether to enable temporal cross-attention in all blocks.
    """

    def __init__(
        self,
        dim,
        depth,
        heads=8,
        dim_head=64,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        use_rel_pos_spatial=True,
        use_rel_pos_temporal=True,
        use_peg_spatial_layers=None,
        use_peg_temporal_layers=None,
        dim_context=None,
        attn_num_null_kv=2,
        use_cross_attn_spatial=True,
        use_cross_attn_temporal=True
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            FactorizedSpatioTemporalBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                causal_temporal=True,
                use_rel_pos_spatial=use_rel_pos_spatial,
                use_rel_pos_temporal=use_rel_pos_temporal,
                use_peg_spatial=(i in (use_peg_spatial_layers or [])),
                use_peg_temporal=(i in (use_peg_temporal_layers or [])),
                use_cross_attn_spatial=use_cross_attn_spatial,
                use_cross_attn_temporal=use_cross_attn_temporal,
                dim_context=dim_context,
                attn_num_null_kv=attn_num_null_kv
            ) for i in range(depth)
        ])

        self.norm = LayerNorm(dim)

    def forward(self, S, context, temporal_mask=None, context_mask=None):
        """
        Args:
            S (Tensor): Decoder input tensor of shape (B, T, H, W, C).
                Represents the spatial-temporal tokens to be autoregressively decoded.

            context (Tensor): Encoder output tensor of shape (B, T_enc, H, W, C).
                This is the video representation that the decoder cross-attends to.

            temporal_mask (BoolTensor, optional): Mask of shape (B, T) indicating
                valid (True) vs padded (False) tokens in decoder input `S`.
                This mask is applied during causal temporal attention.

            context_mask (BoolTensor, optional): Mask for the encoder tokens of shape (B, T_enc).
                Used to mask out padded positions in the encoder output during cross-attention.

        Returns:
            Tensor: Output tensor of shape (B, T, H, W, C) representing the decoded representation.
        """
        # S: (B, T, H, W, C)
        for block in self.blocks:
            S = block(S, context=context, temporal_mask=temporal_mask, context_mask=context_mask)
        return self.norm(S)