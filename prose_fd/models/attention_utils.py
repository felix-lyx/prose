"""
This file contains attention layers and related utils. 
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable, Tuple
from torch import Tensor

from rotary_embedding_torch import RotaryEmbedding

N_MAX_POSITIONS = 1024  # maximum input sequence length

"""
--------------- Attention Variants ---------------
"""


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.batch_first = True

        self.linear_q = nn.Linear(embed_dim, embed_dim, bias)
        self.linear_k = nn.Linear(embed_dim, embed_dim, bias)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        is_causal=False,
        need_weights=False,
        rotary_emb=None,
    ):
        bs, seq_len, _ = query.size()
        k_len = key.size(1)

        # compute projections
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        # split heads (bs, seq_len, dim) -> (bs, n_heads, seq_len, head_dim)
        q = q.view(bs, seq_len, self.num_heads, self.head_dim)
        k = k.view(bs, k_len, self.num_heads, self.head_dim)
        v = v.view(bs, k_len, self.num_heads, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (bs, n_head, seq_len, head_dim)

        if rotary_emb is not None:
            q = rotary_emb.rotate_queries_or_keys(q)
            k = rotary_emb.rotate_queries_or_keys(k)

        # process and merge masks
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                bs,
                k_len,
            ), f"expecting key_padding_mask shape of {(bs, k_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bs, 1, 1, k_len).expand(-1, self.num_heads, -1, -1)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        dropout_p = 0.0 if not self.training else self.dropout

        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, is_causal
        )  # (bs, n_heads, seq_len, head_dim)

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.out_proj(output), None


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Custom implementation of pytorch's TransformerEncoderLayer
    Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        rotary=False,
        custom_attn=False,
        norm=nn.LayerNorm,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(nn.TransformerEncoderLayer, self).__init__()
        if custom_attn:
            assert batch_first
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias)
            self.rotary = rotary
        else:
            self.self_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first, **factory_kwargs
            )
            self.rotary = False

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first

        # self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        # self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm1 = norm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = norm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        rotary_emb=None,
    ) -> Tensor:
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )
        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal, rotary_emb=rotary_emb
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal, rotary_emb=rotary_emb)
            )
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
        rotary_emb=None,
    ) -> Tensor:
        if self.rotary:
            x = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                rotary_emb=rotary_emb,
            )[0]
        else:
            x = self.self_attn(
                x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False, is_causal=is_causal
            )[0]
        return self.dropout1(x)


class CustomTransformerEncoder(nn.Module):
    """
    Custom implementation of pytorch's TransformerEncoder
    Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoder
    """

    def __init__(
        self,
        encoder_layer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
        config=None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.mask_check = mask_check

        if config is not None and config.rotary:
            self.rotary_emb = RotaryEmbedding(dim=config.dim_emb // config.n_head // 2)
            self.rotary = True
        else:
            self.rotary_emb = None
            self.rotary = False

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal: Optional[bool] = None):
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )
        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        batch_first = self.layers[0].self_attn.batch_first
        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        output = src
        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask,
                rotary_emb=self.rotary_emb,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class CausalTransformerDecoder(nn.TransformerDecoder):
    """
    Decoder attention (in encoder-decoder transformer) which supports kv-caching during evaluation.

    The complexity goes from seq_len^3 to seq_len^2.

    In training mode, teacher forcing makes these optimizations unnecessary. Hence the
    Decoder acts like a regular nn.TransformerDecoder (except that the attention tgt
    masks are handled for you).

    Source: https://github.com/alex-matton/causal-transformer-decoder/
    """

    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
        cache: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt (Tensor): current_len_output x bsz x hidden_dim
            memory (Tensor): len_encoded_seq x bsz x hidden_dim
            cache (Optional[Tensor]):
                n_layers x (current_len_output - 1) x bsz x hidden_dim
                If current_len_output == 1, nothing is cached yet, so cache
                should be None. Same if the module is in training mode.
            tgt_mask: assumed to be a causal mask and will be ignored.
            others (Optional[Tensor]): see official documentations
        Returns:
            output (Tensor): current_len_output x bsz x hidden_dim
            cache (Optional[Tensor]): n_layers x current_len_output x bsz x hidden_dim
                Only returns it when module is in eval mode (no caching in training)
        """

        output = tgt

        if self.training:
            if cache is not None:
                raise ValueError("cache parameter should be None in training mode")
            for mod in self.layers:
                output = mod(
                    output,
                    memory,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )

            if self.norm is not None:
                output = self.norm(output)

            return output

        if cache is None:
            assert tgt.size(0) == 1

        new_token_cache = []
        for i, mod in enumerate(self.layers):
            output = mod(output, memory)
            new_token_cache.append(output)
            if cache is not None:
                output = torch.cat([cache[i], output], dim=0)

        if cache is not None:
            new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=1)
        else:
            new_cache = torch.stack(new_token_cache, dim=0)

        if self.norm is not None:
            output = self.norm(output)

        return output, new_cache


class CausalTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Encoder-decoder attention layer.

    Source: https://github.com/alex-matton/causal-transformer-decoder/
    """

    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            see CausalTransformerDecoder
        Returns:
            Tensor:
                If training: embedding of the whole layer: seq_len x bsz x hidden_dim
                If eval mode: embedding of last token: 1 x bsz x hidden_dim
        """

        if self.training:
            return super().forward(
                tgt,
                memory,
                tgt_mask=nn.Transformer.generate_square_subsequent_mask(tgt.size(0), tgt.device),
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        # This part is adapted from the official Pytorch implementation
        # So that only the last token gets modified and returned.
        if self.norm_first:
            tgt_last_tok = tgt[-1:, :, :]

            tgt = self.norm1(tgt)

            # self attention part
            tmp_tgt = self.self_attn(
                tgt[-1:, :, :],
                tgt,
                tgt,
                attn_mask=None,  # not needed because we only care about the last token
                key_padding_mask=tgt_key_padding_mask,
            )[0]
            tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)

            # encoder-decoder attention
            if memory is not None:
                tgt_last_tok = tgt_last_tok + self._mha_block(
                    self.norm2(tgt_last_tok), memory, memory_mask, memory_key_padding_mask
                )

            # final feed-forward network
            tgt_last_tok = tgt_last_tok + self._ff_block(self.norm3(tgt_last_tok))
        else:
            tgt_last_tok = tgt[-1:, :, :]

            # self attention part
            tmp_tgt = self.self_attn(
                tgt_last_tok,
                tgt,
                tgt,
                attn_mask=None,  # not needed because we only care about the last token
                key_padding_mask=tgt_key_padding_mask,
            )[0]
            tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
            tgt_last_tok = self.norm1(tgt_last_tok)

            # encoder-decoder attention
            if memory is not None:
                tgt_last_tok = self.norm2(
                    tgt_last_tok + self._mha_block(tgt_last_tok, memory, memory_mask, memory_key_padding_mask)
                )

            # final feed-forward network
            tgt_last_tok = self.norm3(tgt_last_tok + self._ff_block(tgt_last_tok))
        return tgt_last_tok


class CausalTransformerEncoder(nn.TransformerEncoder):
    """
    Decoder-only attention which supports kv-caching during evaluation.

    The complexity goes from seq_len^3 to seq_len^2.

    """

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
        cache: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            src (Tensor): current_len_output x bsz x hidden_dim
            cache (Optional[Tensor]):
                n_layers x (current_len_output - 1) x bsz x hidden_dim
                If current_len_output == 1, nothing is cached yet, so cache
                should be None. Same if the module is in training mode.
            src_mask: assumed to be a causal mask and will be ignored.
            others (Optional[Tensor]): see official documentations
        Returns:
            output (Tensor): current_len_output x bsz x hidden_dim
            cache (Optional[Tensor]): n_layers x current_len_output x bsz x hidden_dim
                Only returns it when module is in eval mode (no caching in training)
        """

        if self.training:
            if cache is not None:
                raise ValueError("Cache should be None during training")

            return super().forward(
                src=src,
                mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )

        output = src

        new_token_cache = []
        if cache is None:
            # cache everything for the start
            for i, mod in enumerate(self.layers):
                output = mod(src=output, first=True)
                new_token_cache.append(output)
        else:
            # only cache the last token
            for i, mod in enumerate(self.layers):
                output = mod(src=output)
                new_token_cache.append(output)
                if cache is not None:
                    output = torch.cat([cache[i], output], dim=0)

        if cache is not None:
            new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=1)
        else:
            new_cache = torch.stack(new_token_cache, dim=0)

        if self.norm is not None:
            output = self.norm(output)

        return output, new_cache


class CausalDecoderOnlyLayer(nn.TransformerEncoderLayer):
    """
    Decoder-only attention layer.

    Source: https://github.com/alex-matton/causal-transformer-decoder/
    """

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        first: bool = False,
    ) -> Tensor:
        """
        Args:
            see CausalTransformerEncoder
        Returns:
            Tensor:
                If training: embedding of the whole layer: seq_len x bsz x hidden_dim
                If eval mode: embedding of last token: 1 x bsz x hidden_dim
        """

        if self.training:

            return super().forward(
                src=src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )

        # This part is adapted from the official Pytorch implementation
        # So that only the last token gets modified and returned.
        if self.norm_first:
            if first:
                input = src
                src = self.norm1(src)

                # self attention part
                tmp_src = self.self_attn(
                    src,
                    src,
                    src,
                    attn_mask=nn.Transformer.generate_square_subsequent_mask(src.size(0), src.device),
                    key_padding_mask=src_key_padding_mask,
                    need_weights=False,
                    is_causal=True,
                )[0]
                src_last_tok = input + self.dropout1(tmp_src)
            else:
                input = src[-1:, :, :]
                src = self.norm1(src)
                # self attention part
                tmp_src = self.self_attn(
                    src[-1:, :, :],
                    src,
                    src,
                    attn_mask=None,  # not needed because we only care about the last token
                    key_padding_mask=src_key_padding_mask,
                    need_weights=False,
                    is_causal=is_causal,
                )[0]
                src_last_tok = input + self.dropout1(tmp_src)

            # final feed-forward network
            src_last_tok = src_last_tok + self._ff_block(self.norm2(src_last_tok))
        else:
            if first:
                src_last_tok = src
                src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(0), src.device)
                is_causal = True
            else:
                src_last_tok = src[-1:, :, :]
                src_mask = None

            # self attention part
            tmp_src = self.self_attn(
                src_last_tok,
                src,
                src,
                attn_mask=src_mask,  # not needed because we only care about the last token
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )[0]
            src_last_tok = src_last_tok + self.dropout1(tmp_src)
            src_last_tok = self.norm1(src_last_tok)

            # final feed-forward network
            src_last_tok = self.norm2(src_last_tok + self._ff_block(src_last_tok))

        return src_last_tok


class A:
    pass


class OperatorDecoderLayer(nn.TransformerDecoderLayer):
    """OperatorDecoderLayer is made up of multi-head-attn and feedforward network.
    (It is the usual encoder-decoder attention without the self-attention layers)

    Check https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html for details
    Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoderLayer
    """

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        rotary=False,
        custom_attn=False,
        norm=nn.LayerNorm,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(nn.TransformerDecoderLayer, self).__init__()
        if custom_attn:
            assert batch_first
            self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias)
            self.rotary = rotary
            assert not rotary, "currently not supported for operator layers"
        else:
            self.multihead_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias, **factory_kwargs
            )
            self.rotary = False

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first

        # self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        # self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm1 = norm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = norm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        # small hack to handle pytorch TransformerDecoder
        self.self_attn = A()
        self.self_attn.batch_first = batch_first

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        rotary_emb=None,
    ) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            memory_mask: the mask for the memory sequence (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

            tgt_mask, tgt_key_padding_mask, tgt_is_causal: NOT needed as there is not
                self-attention, and will be ignored.

        Shape:
            see the docs in Transformer class.
        """

        x = tgt
        if self.norm_first:
            x = x + self._mha_block(self.norm1(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal, rotary_emb=rotary_emb)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal, rotary_emb=rotary_emb))
            x = self.norm2(x + self._ff_block(x))

        return x

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
        rotary_emb=None,
    ) -> Tensor:
        if self.rotary:
            x = self.multihead_attn(
                x,
                mem,
                mem,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                rotary_emb=rotary_emb,
            )[0]
        else:
            x = self.multihead_attn(
                x,
                mem,
                mem,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                need_weights=False,
            )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


"""
--------------- Positional Embeddings ---------------
"""


class SinusoidalPE(nn.Module):
    """
    Sinusoidal positional embedding.
    Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = N_MAX_POSITIONS):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor, batch_first: bool = True) -> Tensor:
        """
        Arguments:
            x: Tensor [batch_size, seq_len, embedding_dim] if batch_first
                      [seq_len, batch_size, embedding_dim] otherwise
        """

        if batch_first:
            x = x + self.pe[: x.size(1)].transpose(0, 1)
        else:
            x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class LearnablePE(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = N_MAX_POSITIONS):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = Embedding(max_len, d_model)

    def forward(self, x: Tensor, positions: Optional[Tensor] = None, batch_first: bool = True) -> Tensor:
        """
        Arguments:
            x: Tensor [batch_size, seq_len, embedding_dim] if batch_first
                      [seq_len, batch_size, embedding_dim] otherwise
            positions: Tensor [batch_size, seq_len]
        """
        seq_len = x.size(1) if batch_first else x.size(0)
        if positions is None:
            positions = x.new(seq_len).long()
            positions = torch.arange(seq_len, out=positions).unsqueeze(0)  # (1, seq_len)

        pe = self.pe(positions)  # (1, seq_len, d_model)
        if batch_first:
            x = x + pe.expand_as(x)
        else:
            x = x + pe.transpose(0, 1).expand_as(x)

        return self.dropout(x)


def get_embeddings(size, type=None):
    if type is None:
        patch_embeddings = nn.Parameter(torch.randn(*size))
    elif type == "normalize":
        dim = size[-1]
        patch_embeddings = nn.Parameter((dim**-0.5) * torch.randn(*size))
    elif type == "bert":
        patch_embeddings = nn.Parameter(torch.empty(*size).normal_(std=0.02))
    else:
        raise ValueError(f"Unknown type for embedding: {type}")
    return patch_embeddings


"""
--------------- Helper functions ---------------
"""


def get_padding_mask(lengths, max_len=None):
    """
    Input:
        lengths:           LongTensor (bs, )  length of each example
        max_len:           Optional[int]      if None, max_len = lengths.max()
    Output:
        key_padding_mask:  BoolTensor (bs, max_len)    (positions with value True are padding)
    """
    if max_len is None:
        max_len = lengths.max().item()

    bs = lengths.size(0)
    key_padding_mask = torch.arange(max_len, device=lengths.device).expand(bs, max_len) >= lengths.unsqueeze(1)
    return key_padding_mask


def get_block_attn_mask(block_size: int, n_repeat: int, device=torch.device("cpu")):
    """
    Output:
        attn_mask: BoolTensor (block_size * n_repeat, block_size * n_repeat) block diagonal matrix with identity blocks
    """
    blocks = [torch.ones(block_size, block_size, device=device)] * n_repeat
    return torch.block_diag(*blocks).bool()


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_seq_len(src: Tensor, batch_first: bool) -> Optional[int]:
    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _generate_square_subsequent_mask(
    sz: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )


def _detect_is_causal_mask(
    mask: Optional[Tensor],
    is_causal: Optional[bool] = None,
    size: Optional[int] = None,
) -> bool:
    # Prevent type refinement
    make_causal = is_causal is True

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


class GroupNorm(nn.Module):
    """
    Group norm for sequence. Expects input shape (bs, seq_len, d) and splits group only in the last dimension.
    """

    def __init__(self, num_groups, num_channels, affine=True, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_channels))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_channels))

    def forward(self, x):
        bs, seq_len, d = x.size()
        x = x.view(bs, seq_len, self.num_groups, d // self.num_groups)

        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)

        x = x.view(bs, seq_len, d)

        if self.affine:
            x = x * self.weight + self.bias

        return x

    def extra_repr(self) -> str:
        return "{num_groups}, {num_channels}, eps={eps}, affine={affine}".format(**self.__dict__)
