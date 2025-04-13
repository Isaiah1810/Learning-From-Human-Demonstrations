import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from beartype import beartype
from typing import Tuple

from einops import rearrange, repeat

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


def l2norm(t):
    return F.normalize(t, dim=-1)


# bias-less layernorm, being used in more recent T5s, PaLM, also in @borisdayma 's experiments shared with me
# greater stability


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


# feedforward


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4, dropout=0.0):
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


# PEG - position generating module


class PEG(nn.Module):
    def __init__(self, dim, causal=False):
        super().__init__()
        self.causal = causal
        self.dsconv = nn.Conv3d(dim, dim, 3, groups=dim)

    @beartype
    def forward(self, x, shape: Tuple[int, int, int, int] = None):
        needs_shape = x.ndim == 3
        assert not (needs_shape and not exists(shape))

        orig_shape = x.shape

        if needs_shape:
            x = x.reshape(*shape, -1)

        x = rearrange(x, "b ... d -> b d ...")

        frame_padding = (2, 0) if self.causal else (1, 1)

        x = F.pad(x, (1, 1, 1, 1, *frame_padding), value=0.0)
        x = self.dsconv(x)

        x = rearrange(x, "b d ... -> b ... d")

        if needs_shape:
            x = rearrange(x, "b ... d -> b (...) d")

        return x.reshape(orig_shape)


# attention


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context=None,
        dim_head=64,
        heads=8,
        causal=False,
        num_null_kv=0,
        norm_context=True,
        dropout=0.0,
        scale=8,
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = scale
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        if causal:
            self.rel_pos_bias = AlibiPositionalBias(heads=heads)

        self.attn_dropout = nn.Dropout(dropout)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, mask=None, context=None, attn_bias=None):
        batch, device, dtype = x.shape[0], x.device, x.dtype

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))

        if self.null_kv.ndim != 3:
            self.null_kv = self.null_kv.reshape(self.heads, 2 * self.num_null_kv, self.dim_head)
        nk, nv = repeat(self.null_kv, "h (n r) d -> b h n r d", b=batch, r=2).unbind(dim=-2)

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        i, j = sim.shape[-2:]

        if exists(attn_bias):
            attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value=0.0)
            sim = sim + attn_bias

        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            sim = sim + self.rel_pos_bias(sim)

            causal_mask = torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# alibi positional bias for extrapolation


class AlibiPositionalBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, "h -> h 1 1")
        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device=device)
        j_arange = torch.arange(j, device=device)
        bias = -torch.abs(rearrange(j_arange, "j -> 1 1 j") - rearrange(i_arange, "i -> 1 i 1"))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: heads - closest_power_of_2]
        )

    def forward(self, sim):
        h, i, j, device = *sim.shape[-3:], sim.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer("bias", bias, persistent=False)

        return self.bias


class ContinuousPositionBias(nn.Module):
    """from https://arxiv.org/abs/2111.09883"""

    def __init__(
        self, *, dim, heads, num_dims=2, layers=2, log_dist=True, cache_rel_pos=False  # 2 for images, 3 for video
    ):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist

        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), leaky_relu()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))

        self.net.append(nn.Linear(dim, heads))

        self.cache_rel_pos = cache_rel_pos
        self.register_buffer("rel_pos", None, persistent=False)

    def forward(self, *dimensions, device=torch.device("cpu")):

        if not exists(self.rel_pos) or not self.cache_rel_pos:
            positions = [torch.arange(d, device=device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing="ij"))
            grid = rearrange(grid, "c ... -> (...) c")
            rel_pos = rearrange(grid, "i c -> i 1 c") - rearrange(grid, "j c -> 1 j c")

            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

            self.register_buffer("rel_pos", rel_pos, persistent=False)

        rel_pos = self.rel_pos.float()

        for layer in self.net:
            rel_pos = layer(rel_pos)

        return rearrange(rel_pos, "i j h -> h i j")


# transformer


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_context=None,
        causal=False,
        dim_head=64,
        heads=8,
        ff_mult=4,
        peg=False,
        peg_causal=False,
        attn_num_null_kv=2,
        has_cross_attn=False,
        attn_dropout=0.0,
        ff_dropout=0.0
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PEG(dim=dim, causal=peg_causal) if peg else None,
                        Attention(dim=dim, dim_head=dim_head, heads=heads, causal=causal, dropout=attn_dropout),
                        (
                            Attention(
                                dim=dim,
                                dim_head=dim_head,
                                dim_context=dim_context,
                                heads=heads,
                                causal=False,
                                num_null_kv=attn_num_null_kv,
                                dropout=attn_dropout,
                            )
                            if has_cross_attn
                            else None
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.norm_out = LayerNorm(dim)

    @beartype
    def forward(
        self,
        x,
        video_shape: Tuple[int, int, int, int] = None,
        attn_bias=None,
        context=None,
        self_attn_mask=None,
        cross_attn_context_mask=None,
    ):

        for peg, self_attn, cross_attn, ff in self.layers:
            if exists(peg):
                x = peg(x, shape=video_shape) + x

            x = self_attn(x, attn_bias=attn_bias, mask=self_attn_mask) + x

            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context=context, mask=cross_attn_context_mask) + x

            x = ff(x) + x

        return self.norm_out(x)


class FactorizedSpatioTemporalBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        causal_temporal=False,
        use_rel_pos_spatial=False,
        use_rel_pos_temporal=False,
        use_peg_spatial=False,
        use_peg_temporal=False,
        use_cross_attn_spatial=False,
        use_cross_attn_temporal=False,
        dim_context=None,
        attn_num_null_kv=2
    ):
        super().__init__()

        self.use_peg_spatial = use_peg_spatial
        self.use_peg_temporal = use_peg_temporal
        self.use_cross_attn_spatial = use_cross_attn_spatial
        self.use_cross_attn_temporal = use_cross_attn_temporal
        self.causal_temporal = causal_temporal

        self.peg_spatial = PEG(dim=dim, causal=False) if use_peg_spatial else None
        self.peg_temporal = PEG(dim=dim, causal=causal_temporal) if use_peg_temporal else None

        # --- Spatial ---
        self.spatial_attn = Attention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            causal=False
        )

        self.rel_pos_spatial = ContinuousPositionBias(
            dim=dim_head,
            heads=heads,
            num_dims=2
        ) if use_rel_pos_spatial else None

        self.cross_attn_spatial = Attention(
            dim=dim,
            dim_context=dim_context or dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            num_null_kv=attn_num_null_kv
        ) if use_cross_attn_spatial else None

        # --- Temporal ---
        self.temporal_attn = Attention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            causal=causal_temporal
        )

        self.rel_pos_temporal = ContinuousPositionBias(
            dim=dim_head,
            heads=heads,
            num_dims=1
        ) if use_rel_pos_temporal else None

        self.cross_attn_temporal = Attention(
            dim=dim,
            dim_context=dim_context or dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            num_null_kv=attn_num_null_kv
        ) if use_cross_attn_temporal else None

        # --- Feedforward + Norm ---
        self.ff = FeedForward(dim, mult=ff_mult, dropout=ff_dropout)
        self.norm = LayerNorm(dim)

    def forward(self, x, context=None, context_mask=None, temporal_mask=None):
        # x: (B, T, H, W, C)
        b, t, h, w, c = x.shape

        # ---------- Spatial ----------
        x_spatial = rearrange(x, 'b t h w c -> (b t) h w c')
        if exists(self.peg_spatial):
            x_spatial = self.peg_spatial(x_spatial, shape=(b * t, h, w)) + x_spatial
        x_spatial = rearrange(x_spatial, '(b t) h w c -> (b t) (h w) c', b=b, t=t)

        attn_bias_spatial = self.rel_pos_spatial(h, w, device=x.device) if self.rel_pos_spatial else None
        x_spatial = self.spatial_attn(x_spatial, attn_bias=attn_bias_spatial) + x_spatial

        if self.use_cross_attn_spatial and context is not None:
            context_spatial = rearrange(context, 'b t h w c -> (b t) (h w) c')
            context_mask_spatial = repeat(context_mask, 'b t -> (b t) (h w)', h=h, w=w) if context_mask is not None else None
            x_spatial = self.cross_attn_spatial(x_spatial, context=context_spatial, mask=context_mask_spatial) + x_spatial

        x = rearrange(x_spatial, '(b t) (h w) c -> b t h w c', b=b, t=t, h=h, w=w)

        # ---------- Temporal ----------
        x_temporal = rearrange(x, 'b t h w c -> (b h w) t c')
        if exists(self.peg_temporal):
            x_temporal = self.peg_temporal(x_temporal, shape=(b, h, w, t)) + x_temporal

        temporal_mask = temporal_mask = repeat(temporal_mask, 'b t -> (b h w) t', h=h, w=w) if temporal_mask is not None else None
        attn_bias_temporal = self.rel_pos_temporal(t, device=x.device) if self.rel_pos_temporal else None
        x_temporal = self.temporal_attn(x_temporal, attn_bias=attn_bias_temporal, mask=temporal_mask) + x_temporal

        if self.use_cross_attn_temporal and context is not None:
            context_temporal = rearrange(context, 'b t h w c -> (b h w) t c')
            context_mask_temporal = repeat(context_mask, 'b t -> (b h w) t', h=h, w=w) if context_mask is not None else None
            x_temporal = self.cross_attn_temporal(x_temporal, context=context_temporal, mask=context_mask_temporal) + x_temporal

        x = rearrange(x_temporal, '(b h w) t c -> b t h w c', b=b, h=h, w=w)

        # ---------- FF + Norm ----------
        x = self.ff(x) + x
        return self.norm(x)


class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads=8,
        dim_head=64,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        causal_temporal=False,
        use_rel_pos_spatial=False,
        use_rel_pos_temporal=False,
        use_peg_spatial_layers=None,
        use_peg_temporal_layers=None,
        use_cross_attn_spatial_layers=None,
        use_cross_attn_temporal_layers=None,
        dim_context=None,
        attn_num_null_kv=2
    ):
        super().__init__()

        use_peg_spatial_layers = use_peg_spatial_layers or []
        use_peg_temporal_layers = use_peg_temporal_layers or []
        use_cross_attn_spatial_layers = use_cross_attn_spatial_layers or []
        use_cross_attn_temporal_layers = use_cross_attn_temporal_layers or []

        self.blocks = nn.ModuleList([
            FactorizedSpatioTemporalBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                causal_temporal=causal_temporal,
                use_rel_pos_spatial=use_rel_pos_spatial,
                use_rel_pos_temporal=use_rel_pos_temporal,
                use_peg_spatial=(i in use_peg_spatial_layers),
                use_peg_temporal=(i in use_peg_temporal_layers),
                use_cross_attn_spatial=(i in use_cross_attn_spatial_layers),
                use_cross_attn_temporal=(i in use_cross_attn_temporal_layers),
                dim_context=dim_context,
                attn_num_null_kv=attn_num_null_kv
            ) for i in range(depth)
        ])

        self.norm = LayerNorm(dim)

    def forward(self, x, context=None, context_mask=None, temporal_mask=None):
        for block in self.blocks:
            x = block(x, context=context, context_mask=context_mask, temporal_mask=temporal_mask)
        return self.norm(x)
