from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum, nn


def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value=False)

    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out

    return inner


@dataclass
class LayerIntermediates:
    last_hidden: Optional[Tensor] = None  # very last hidden after all attention layers, after the final norm
    cached_kvs: Optional[List[Tensor]] = None


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim**-0.5
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device)
        return self.emb(pos) * self.scale


class LayerNorm(nn.Module):
    def __init__(self, dim):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class FeedForward(nn.Module):
    def __init__(self, dim, inner_dim, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=True),
        )

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def create_causal_mask(self, i, j, device):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

    def forward(self, x, return_cached_kv=False, cache=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        h = self.heads

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h)
        k = rearrange(self.to_k(x), "b n (h d) -> b h n d", h=h)
        v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=h)

        if cache is not None:
            ck, cv = cache
            k = torch.cat((ck, k), dim=-2)
            v = torch.cat((cv, v), dim=-2)

        if return_cached_kv:
            cached_kv = (k, v)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * (q.shape[-1] ** -0.5)

        if q.shape[-2] != 1:
            causal_mask = self.create_causal_mask(dots.shape[-2], dots.shape[-1], device=q.device)
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(causal_mask, mask_value)

        attn = F.softmax(dots, dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        if not return_cached_kv:
            return out

        return out, cached_kv


class DecTransformerLayer(nn.Module):
    def __init__(self, dim, heads, attn_dropout, ff_inner_dim, ff_dropout):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, dropout=attn_dropout)
        self.norm2 = LayerNorm(dim)
        self.ff = FeedForward(dim, inner_dim=ff_inner_dim, dropout=ff_dropout)

    def forward(self, x, cache=None, return_cached_kv=False):
        x = self.norm1(x)
        x, attn_inter = self.attn(x, cache=cache, return_cached_kv=return_cached_kv)
        x = self.norm2(x)
        x = self.ff(x)
        return x, attn_inter


class DecTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        num_tokens,
        max_seq_len,
        emb_dropout,
        attn_dropout,
        ff_inner_dim,
        ff_dropout,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList(
            [DecTransformerLayer(dim, heads, attn_dropout, ff_inner_dim, ff_dropout) for _ in range(depth)]
        )
        self.final_norm = LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        nn.init.kaiming_normal_(self.token_emb.weight)

    def forward(self, x, return_intermediates=False, cache=None):
        # absolute positional embedding
        x = self.token_emb(x) + self.pos_emb(x)
        x = self.emb_dropout(x)

        # assume cached key / values
        cached_kvs = []
        if cache is not None:
            x = x[:, -1:]
            cached_kvs = cache.cached_kvs

        iter_attn_cache = iter(cached_kvs)

        for layer in self.layers:
            x, cached_kv = layer(x, cache=next(iter_attn_cache, None), return_cached_kv=True)

            if return_intermediates:
                cached_kvs.append(cached_kv)

        x = self.final_norm(x)

        intermediates = LayerIntermediates(cached_kvs=cached_kvs)

        # project to logits
        logits = self.to_logits(x)

        if return_intermediates:
            return logits, intermediates
        return logits


class WrapperDecTransformer(nn.Module):
    def __init__(self, ignore_index=-100, pad_value=0, **net_kwargs):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.net = DecTransformer(**net_kwargs)
        self.sampling_func = top_p

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prompts,
        seq_len,
        eos_token,
    ):
        out = prompts
        cache = None

        for _ in range(seq_len):
            logits, new_cache = self.net(out, return_intermediates=True, cache=cache)
            cache = new_cache

            logits = logits[:, -1]
            filtered_logits = self.sampling_func(logits)
            probs = F.softmax(filtered_logits, dim=-1)
            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            is_eos_tokens = out == eos_token

            if is_eos_tokens.any(dim=-1).all():
                break

        shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
        mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
        out = out.masked_fill(mask, self.pad_value)

        return out

    def forward(self, x, return_outputs=False):
        ignore_index = self.ignore_index

        inp, target = x[:, :-1], x[:, 1:]
        inp = torch.where(inp == ignore_index, self.pad_value, inp)

        logits, cache = self.net(inp, return_intermediates=True)
        loss = F.cross_entropy(rearrange(logits, "b n c -> b c n"), target, ignore_index=ignore_index)

        if return_outputs:
            return loss, (logits, cache)
        return loss


if __name__ == "__main__":
    model = WrapperDecTransformer(
        dim=512,
        depth=6,
        heads=8,
        num_tokens=10000,
        max_seq_len=2048,
        emb_dropout=0.1,
        attn_dropout=0.1,
        ff_inner_dim=2048,
        ff_dropout=0.1,
    )
    print(model)
    x = torch.randint(0, 10000, (1, 1024))
    print(model(x))
    x = torch.randint(0, 10000, (1, 10))
    print(model.generate(x, 10, 2))
