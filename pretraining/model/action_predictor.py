from torch import nn
import torch
import torch.nn.functional as F
from .encoder import VideoEncoder
from .decoder import VideoDecoder
from sequence_tokenizer import SequenceTokenizer

def MLPProjector(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim)
    )


class VideoToAction(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim,
        action_dim,
        encoder_depth,
        decoder_depth,
        mlp_hidden_dim=2048,
        heads=8,
        dim_head=64,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        use_rel_pos_spatial=True,
        use_rel_pos_temporal=True,
        use_peg_spatial_layers_enc=None,
        use_peg_temporal_layers_enc=None,
        use_peg_spatial_layers_dec=None,
        use_peg_temporal_layers_dec=None,
        attn_num_null_kv=2,
        loss_type='l2',
        tokenizer_config=None,
        use_tokenizer=True
    ):
        super().__init__()

        self.encoder_input_proj = MLPProjector(input_dim, mlp_hidden_dim, model_dim)
        self.decoder_input_proj = MLPProjector(input_dim, mlp_hidden_dim, model_dim)
        self.output_proj = nn.Linear(model_dim, action_dim)

        self.loss_type = loss_type

        self.encoder = VideoEncoder(
            dim=model_dim,
            depth=encoder_depth,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            use_rel_pos_spatial=use_rel_pos_spatial,
            use_rel_pos_temporal=use_rel_pos_temporal,
            use_peg_spatial_layers=use_peg_spatial_layers_enc,
            use_peg_temporal_layers=use_peg_temporal_layers_enc
        )

        self.decoder = VideoDecoder(
            dim=model_dim,
            depth=decoder_depth,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            use_rel_pos_spatial=use_rel_pos_spatial,
            use_rel_pos_temporal=use_rel_pos_temporal,
            use_peg_spatial_layers=use_peg_spatial_layers_dec,
            use_peg_temporal_layers=use_peg_temporal_layers_dec,
            dim_context=model_dim,
            attn_num_null_kv=attn_num_null_kv,
            use_cross_attn_spatial=True,
            use_cross_attn_temporal=True
        )
        if use_tokenizer:
            self.tokenizer = SequenceTokenizer(tokenizer_config, 'cuda')
        else:
            self.tokenizer = None

    def forward(self, V, S, A=None, temporal_mask_V=None, temporal_mask_S=None, context_mask=None, return_loss=True):
        """
        Args:
            V (Tensor): Encoder input of shape (B, T, H, W, input_dim)
            S (Tensor): Decoder input of shape (B, T, H, W, input_dim)
            A (Tensor, optional): Ground truth actions of shape (B, T, H, W, action_dim). Required if return_loss=True
            temporal_mask_V (BoolTensor, optional): Padding mask for encoder input
            temporal_mask_S (BoolTensor, optional): Padding mask for decoder input
            context_mask (BoolTensor, optional): Mask for encoder output during cross-attn
            return_loss (bool): If True, returns the loss along with predictions

        Returns:
            If return_loss=True: tuple (A_hat, loss)
            If return_loss=False: A_hat only
        """
        if (A == 0).all(): 
            if self.tokenizer is None:
                assert "Tokenizer must be initialized if A is all 0"
            action = self.tokenizer.extract_actions(S)
            action_pad = torch.zeros((action.shape[0], 1, action.shape[2])).to(action.device)
            A = torch.cat((action_pad, action), dim=1).unsqueeze(2).unsqueeze(3)
            A = A.expand(-1, -1, S.shape[2], S.shape[3], -1)
        else:
            raise NotImplementedError

        V = self.encoder_input_proj(V)
        S = self.decoder_input_proj(S)

        enc_out = self.encoder(V, temporal_mask=temporal_mask_V)  # (B, T, H, W, model_dim)
        dec_out = self.decoder(S, context=enc_out, temporal_mask=temporal_mask_S, context_mask=context_mask)

        A_hat = self.output_proj(dec_out)  # predicted actions


        if return_loss:
            assert A is not None, "Ground truth actions (A) must be provided if return_loss=True"
            if self.loss_type == 'l1':
                loss = F.l1_loss(A_hat, A, reduction='mean')
            else:
                loss = F.mse_loss(A_hat, A, reduction='mean')
            return A_hat, loss

        return A_hat
