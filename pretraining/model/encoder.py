import torch.nn as nn
from .attention import SpatioTemporalTransformer

class VideoEncoder(nn.Module):
    """
    A bi-directional transformer encoder for processing spatio-temporal video tokens.

    Args:
        dim (int): Dimensionality of input and output embeddings.
        depth (int): Number of transformer blocks.
        heads (int): Number of attention heads per block.
        dim_head (int): Dimensionality of each attention head.
        ff_mult (int): Feedforward hidden layer multiplier.
        attn_dropout (float): Dropout rate for attention weights.
        ff_dropout (float): Dropout rate in feedforward layers.
        use_rel_pos_spatial (bool): Whether to use relative positional bias in spatial attention.
        use_rel_pos_temporal (bool): Whether to use relative positional bias in temporal attention.
        use_peg_spatial_layers (List[int], optional): Indices of layers to apply PEG in spatial blocks.
        use_peg_temporal_layers (List[int], optional): Indices of layers to apply PEG in temporal blocks.
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
        use_peg_temporal_layers=None
    ):
        super().__init__()

        self.encoder = SpatioTemporalTransformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            causal_temporal=False,  # bi-directional encoder
            use_rel_pos_spatial=use_rel_pos_spatial,
            use_rel_pos_temporal=use_rel_pos_temporal,
            use_peg_spatial_layers=use_peg_spatial_layers,
            use_peg_temporal_layers=use_peg_temporal_layers
        )

    def forward(self, video, temporal_mask=None):
        """
        Args:
            video (Tensor): Input video tensor of shape (B, T, H, W, C),
                where B is the batch size, T is the number of frames,
                H and W are spatial dimensions, and C is the feature dimension.

            temporal_mask (BoolTensor, optional): Padding mask of shape (B, T).
                True indicates a valid timestep, and False indicates a padded timestep.
                This mask is broadcasted across spatial dimensions and only applied
                during the temporal attention stage in each block.

        Returns:
            Tensor: Encoded video features of shape (B, T, H, W, C).
        """
        # video: (B, T, H, W, C)
        return self.encoder(video, temporal_mask=temporal_mask)  # output: (B, T, H, W, C)
