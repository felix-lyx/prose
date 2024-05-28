from logging import getLogger
import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


N_MAX_POSITIONS = 512  # maximum input sequence length
GLOBAL_STORE_OUTPUTS = False

logger = getLogger()


def round_up(x, base=64):
    return base * round(float(x) / base)


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out.detach_()
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))


def get_sliding_window_masks(slen, lengths, window_size):
    """
    Generate attention mask for sliding window prediction
    INPUT: slen = input_len + window_size
           lengths (bs,) input_len + window_size
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    attn_mask = attn_mask ^ torch.tril(attn_mask, diagonal=-window_size)  # (bs, slen, slen)
    attn_mask[:, :window_size, :window_size] = True

    # sanity check
    assert mask.size() == (bs, slen)

    return mask, attn_mask


def get_masks(slen, lengths, causal):
    """
    Generate hidden states mask, and optionally an attention mask.
    Outputs: mask is just based on all lengths,
             if causal, attn_mask will be just lower triangular
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask


def get_fusion_masks(slen, slen_data, slen_text, lengths1, lengths2, causal=False):
    """
    Generate mask for fusion by combining the mask for data and text
    Outputs: mask is just based on all lengths,
             if causal, attn_mask will be just lower triangular
    """
    assert lengths1.max().item() <= slen
    bs = lengths1.size(0)

    alen_data = torch.arange(slen_data, dtype=torch.long, device=lengths1.device)
    mask_data = alen_data < lengths1[:, None]
    assert mask_data.size() == (bs, slen_data)

    alen_text = torch.arange(slen_text, dtype=torch.long, device=lengths2.device)
    mask_text = alen_text < lengths2[:, None]
    assert mask_text.size() == (bs, slen_text)

    if causal:
        raise NotImplementedError
    else:
        mask = torch.cat((mask_data, mask_text), dim=1)
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask


class MultiHeadAttention(nn.Module):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, src_dim, dropout, normalized_attention, use_library=True):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.src_dim = src_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.normalized_attention = normalized_attention
        assert self.dim % self.n_heads == 0

        self.use_library = use_library
        if not self.use_library:
            self.q_lin = nn.Linear(dim, dim)
            self.k_lin = nn.Linear(src_dim, dim)
            self.v_lin = nn.Linear(src_dim, dim)
            if self.normalized_attention:
                self.attention_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(dim // n_heads)))
            self.out_lin = nn.Linear(dim, dim)

        else:
            # use library attention
            self.attn = nn.MultiheadAttention(self.dim, self.n_heads, dropout=self.dropout, batch_first=True)

    def forward(self, input, mask=None, kv=None, causal=False, use_cache=False, save=False, idx=-1):
        """
        Self-attention (if kv is None)
        or attention over source sentence (provided by kv).
            input   (bs, qlen, dim)
            mask    (bs, klen) (non-causal) or (bs, klen, klen)
        """
        assert not (use_cache and self.cache is None)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if not use_cache else self.cache["slen"] + qlen
        else:
            klen = kv.size(1)
        assert dim == self.dim, "Dimensions do not match: %s input vs %s configured" % (
            dim,
            self.dim,
        )

        n_heads = self.n_heads
        dim_per_head = dim // n_heads

        if not self.use_library:
            # use existing code for Multihead Attention

            def shape(x):
                """projection"""
                return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

            def unshape(x):
                """compute context"""
                return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

            q = shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            if kv is None:
                k = shape(self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)
                v = shape(self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            elif not use_cache or self.layer_id not in self.cache:
                k = v = kv
                k = shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
                v = shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

            if use_cache:
                if self.layer_id in self.cache:
                    if kv is None:
                        k_, v_ = self.cache[self.layer_id]
                        k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                        v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
                    else:
                        k, v = self.cache[self.layer_id]
                self.cache[self.layer_id] = (k, v)
            if self.normalized_attention:
                q = F.normalize(q, p=2, dim=-1)
                k = F.normalize(k, p=2, dim=-1)
                q = q * self.attention_scale
            else:
                q = q / math.sqrt(dim_per_head)  # (bs, n_heads, qlen, dim_per_head)

            scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, qlen, klen)

            if mask is not None:
                mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)
                mask = (mask == 0).view(mask_reshape).expand_as(scores)  # (bs, n_heads, qlen, klen)
                scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, qlen, klen)

            weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
            weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)
            context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
            context = unshape(context)  # (bs, qlen, dim)

            if GLOBAL_STORE_OUTPUTS and not self.training:
                self.outputs = weights.detach().cpu()

            return self.out_lin(context)

        else:
            # use Multihead Attention provided by Pytorch
            # input has dimension (bs, qlen, dim)
            # mask is (bs, klen) (non-causal) or (bs, qlen, klen)

            assert not use_cache
            if kv is None:
                kv = input

            key_padding_mask = None
            attn_mask = None

            if mask is not None:
                if mask.dim() == 3:
                    # causal
                    mask_reshape = (bs, 1, qlen, klen)
                    mask = (mask == 0).view(mask_reshape).expand(bs, n_heads, qlen, klen)
                    attn_mask = mask.reshape(bs * n_heads, qlen, klen).bool()
                else:
                    # non-causal
                    key_padding_mask = mask == 0

            # context (bs, qlen, dim), weights (bs, n_heads, qlen, klen)
            context, weights = self.attn(
                input,  # query
                kv,  # key
                kv,  # value
                key_padding_mask=key_padding_mask,
                average_attn_weights=False,
                attn_mask=attn_mask,
            )

            if save:
                array = weights.numpy(force=True)
                logger.info("layer {} shape {}".format(idx, array.shape))
                np.save(f"tmp_save/layer{idx}.npy", array)

            if GLOBAL_STORE_OUTPUTS and not self.training:
                self.outputs = weights.detach().cpu()

            return context


class TransformerFFN(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, hidden_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.midlin = nn.ModuleList()
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        for i in range(1, self.hidden_layers):
            self.midlin.append(nn.Linear(dim_hidden, dim_hidden))
        self.lin2 = nn.Linear(dim_hidden, out_dim)

    def forward(self, input):
        x = self.lin1(input)
        x = F.gelu(x)
        for mlin in self.midlin:
            x = mlin(x)
            x = F.gelu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class FusionTransformerModel(nn.Module):
    STORE_OUTPUTS = GLOBAL_STORE_OUTPUTS

    def __init__(
        self,
        params,
        positional_embeddings,
    ):
        """
        Transformer model for fusion.
        """
        super().__init__()

        # encoder / decoder, output layer
        self.dtype = torch.float

        # model parameters

        self.dim = params.fusion_emb_dim  # 512 by default
        self.src_dim = params.fusion_emb_dim
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_hidden_layers = params.n_fusion_hidden_layers
        self.n_heads = params.n_fusion_heads  # 8 by default
        self.n_layers = params.n_fusion_layers

        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        self.norm_attention = params.norm_attention
        assert self.dim % self.n_heads == 0, "transformer dim must be a multiple of n_heads"

        # embeddings

        if positional_embeddings is None or positional_embeddings == "alibi":
            self.position_embeddings_data = self.position_embeddings_text = None
        elif positional_embeddings == "sinusoidal":
            self.position_embeddings_data = Embedding(N_MAX_POSITIONS, self.dim)
            create_sinusoidal_embeddings(N_MAX_POSITIONS, self.dim, out=self.position_embeddings_data.weight)
            self.position_embeddings_text = Embedding(N_MAX_POSITIONS, self.dim)
            create_sinusoidal_embeddings(N_MAX_POSITIONS, self.dim, out=self.position_embeddings_text.weight)
        elif positional_embeddings == "learnable":
            self.position_embeddings_data = Embedding(N_MAX_POSITIONS, self.dim)
            self.position_embeddings_text = Embedding(N_MAX_POSITIONS, self.dim)
        else:
            raise NotImplementedError

        if params.fusion_type_embeddings:
            self.type_embeddings = Embedding(2, self.dim)
        else:
            self.type_embeddings = None

        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)

        # transformer layers
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()

        for layer_id in range(self.n_layers):
            self.attentions.append(
                MultiHeadAttention(
                    self.n_heads,
                    self.dim,
                    self.dim,
                    dropout=self.attention_dropout,
                    normalized_attention=self.norm_attention,
                    use_library=params.use_library_attention,
                )
            )
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            self.ffns.append(
                TransformerFFN(
                    self.dim,
                    self.hidden_dim,
                    self.dim,
                    self.n_hidden_layers,
                    dropout=self.dropout,
                )
            )
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))

        self.cache = None

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(
        self,
        x_data,
        x_text,
        lengths_data,
        lengths_text,
        causal=False,
        positions_data=None,
        positions_text=None,
        use_cache=False,
    ):
        """
        Inputs:
            x_data        (data_len, bs, dim),
            x_text        (text_len, bs, dim)
            lengths_data  LongTensor(bs), containing the length of each sentence in x_data
            lengths_text  LongTensor(bs), containing the length of each sentence in x_text
        """

        # check input dimensions
        slen_data, bs, dim1 = x_data.size()
        slen_text, bs2, dim2 = x_text.size()
        assert bs == bs2
        assert dim1 == dim2
        assert lengths_data.size(0) == bs
        assert lengths_text.size(0) == bs
        assert lengths_data.max().item() <= slen_data
        assert lengths_text.max().item() <= slen_text

        x_data = x_data.transpose(0, 1)  # batch size as dimension 0, (bs, data_len, dim)
        x_text = x_text.transpose(0, 1)  # (bs, text_len, dim)

        slen = slen_data + slen_text
        assert not (use_cache and self.cache is None)

        # generate masks
        mask, attn_mask = get_fusion_masks(slen, slen_data, slen_text, lengths_data, lengths_text, causal)

        # positions
        if self.position_embeddings_data is not None:
            if positions_data is None:
                positions_data = x_data.new(slen_data).long()
                positions_data = torch.arange(slen_data, out=positions_data).unsqueeze(0)  # (1, slen_data)
            else:
                assert positions_data.size() == (slen_data, bs)
                positions_data = positions_data.transpose(0, 1)
        if self.position_embeddings_text is not None:
            if positions_text is None:
                positions_text = x_text.new(slen_text).long()
                positions_text = torch.arange(slen_text, out=positions_text).unsqueeze(0)
            else:
                assert positions_text.size() == (slen_text, bs)
                positions_text = positions_text.transpose(0, 1)

        # do not recompute cached elements
        # if use_cache:
        #     _slen = slen - self.cache["slen"]
        #     x = x[:, -_slen:]
        #     positions = positions[:, -_slen:]
        #     mask = mask[:, -_slen:]
        #     attn_mask = attn_mask[:, -_slen:]

        # all layer outputs
        if FusionTransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs = []

        if self.position_embeddings_data is not None:
            x_data = x_data + self.position_embeddings_data(positions_data).expand_as(x_data)
        if self.position_embeddings_text is not None:
            x_text = x_text + self.position_embeddings_text(positions_text).expand_as(x_text)

        if self.type_embeddings is not None:
            type_data = torch.zeros(1, 1, dtype=torch.long, device=x_data.device)
            type_text = torch.ones(1, 1, dtype=torch.long, device=x_text.device)

            x_data = x_data + self.type_embeddings(type_data).expand_as(x_data)
            x_text = x_text + self.type_embeddings(type_text).expand_as(x_text)

        tensor = torch.cat((x_data, x_text), dim=1)  # (bs, data_len+text_len, dim)

        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        if FusionTransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs.append(tensor.detach().cpu())

        # transformer layers
        for i in range(self.n_layers):
            # self attention
            self.attentions[i].cache = self.cache
            attn = self.attentions[i](tensor, attn_mask, use_cache=use_cache)
            # attn = self.attentions[i](tensor, attn_mask, use_cache=use_cache, save=True, idx=i)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)

            tensor *= mask.unsqueeze(-1).to(tensor.dtype)
            if FusionTransformerModel.STORE_OUTPUTS and not self.training:
                self.outputs.append(tensor.detach().cpu())

        # update cache length
        # if use_cache:
        #     self.cache["slen"] += tensor.size(1)

        # move back sequence length to dimension 0
        tensor = tensor.transpose(0, 1)

        return tensor


class DataTransformerModel(nn.Module):
    STORE_OUTPUTS = GLOBAL_STORE_OUTPUTS

    def __init__(
        self,
        params,
        is_encoder,
        with_output,
        positional_embeddings,
        # initial_embedder=None,
    ):
        """
        Transformer model (encoder or decoder) for data.
        """
        super().__init__()

        # encoder / decoder, output layer
        self.dtype = torch.float
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output

        # model parameters

        self.dim = params.data_enc_emb_dim if is_encoder else params.data_dec_emb_dim  # 512 by default
        self.src_dim = params.data_enc_emb_dim
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_hidden_layers = params.n_data_enc_hidden_layers if is_encoder else params.n_data_dec_hidden_layers
        self.n_heads = params.n_data_enc_heads if is_encoder else params.n_data_dec_heads  # 8 by default
        self.n_layers = params.n_data_enc_layers if is_encoder else params.n_data_dec_layers

        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        self.norm_attention = params.norm_attention
        assert self.dim % self.n_heads == 0, "transformer dim must be a multiple of n_heads"

        # embeddings

        # if initial_embedder is not None:
        #     self.initial_embedder = initial_embedder
        # else:
        #     self.initial_embedder = nn.Linear(2, self.dim)

        if positional_embeddings is None or positional_embeddings == "alibi":
            self.position_embeddings = None
        elif positional_embeddings == "sinusoidal":
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
            create_sinusoidal_embeddings(N_MAX_POSITIONS, self.dim, out=self.position_embeddings.weight)
        elif positional_embeddings == "learnable":
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        else:
            raise NotImplementedError

        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)

        # transformer layers
        self.use_library_attention = params.use_library_attention
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        if self.is_decoder:
            self.layer_norm15 = nn.ModuleList()
            self.encoder_attn = nn.ModuleList()

        for layer_id in range(self.n_layers):
            self.attentions.append(
                MultiHeadAttention(
                    self.n_heads,
                    self.dim,
                    self.dim,
                    dropout=self.attention_dropout,
                    normalized_attention=self.norm_attention,
                    use_library=params.use_library_attention,
                )
            )
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            if self.is_decoder:
                self.layer_norm15.append(nn.LayerNorm(self.dim, eps=1e-12))
                self.encoder_attn.append(
                    MultiHeadAttention(
                        self.n_heads,
                        self.dim,
                        self.src_dim,
                        dropout=self.attention_dropout,
                        normalized_attention=self.norm_attention,
                        use_library=params.use_library_attention,
                    )
                )
            self.ffns.append(
                TransformerFFN(
                    self.dim,
                    self.hidden_dim,
                    self.dim,
                    self.n_hidden_layers,
                    dropout=self.dropout,
                )
            )
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))

        self.cache = None

        # output layer
        if self.with_output:
            self.proj = nn.Linear(self.dim, 1, bias=True)
        self.split_fused_feature = params.split_fused_feature_data

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "predict":
            return self.predict(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(
        self,
        x,
        lengths,
        causal,
        src_enc=None,
        src_len=None,
        window_size=None,
        positions=None,
        use_cache=False,
    ):
        """
        Inputs:
            x          (slen, bs, emb_dim), containing float time series for encoder
            lengths    LongTensor(bs), containing the length of each sentence
            causal     Boolean, if True, the attention is only done over previous hidden states
            positions  LongTensor(slen, bs), containing word positions
        """

        # check inputs
        slen, bs = x.size()[:2]
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)  # batch size as dimension 0
        assert (src_enc is None) == (src_len is None)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.size(0) == bs
        assert not (use_cache and self.cache is None)

        # generate masks
        if window_size is None:
            mask, attn_mask = get_masks(slen, lengths, causal)
        else:
            mask, attn_mask = get_sliding_window_masks(slen, lengths, window_size)

        if self.is_decoder and src_enc is not None:
            # if use_cache and ('src_mask' in self.cache):
            #     src_mask = self.cache['src_mask']
            #     if self.split_fused_feature:
            #         max_data_len = self.cache['max_data_len']
            #         src_enc = src_enc[:,max_data_len:,:]
            # else:
            src_data_len, src_text_len = src_len[0], src_len[1]
            max_data_len = src_data_len.max()
            src_text_mask = (
                torch.arange(src_text_len.max(), dtype=torch.long, device=lengths.device) < src_text_len[:, None]
            )  # (bs, text_len)

            if self.split_fused_feature:
                src_mask = src_text_mask
                src_enc = src_enc[:, max_data_len:, :]
            else:
                src_data_mask = (
                    torch.arange(max_data_len, dtype=torch.long, device=lengths.device) < src_data_len[:, None]
                )  # (bs, data_len)

                src_mask = torch.cat([src_data_mask, src_text_mask], dim=1)
                # if use_cache:
                #     self.cache['src_mask'] = src_mask
                #     self.cache['max_data_len'] = max_data_len

        # positions
        if self.position_embeddings is not None:
            if positions is None:
                positions = x.new(slen).long()
                positions = torch.arange(slen, out=positions).unsqueeze(0)
            else:
                assert positions.size() == (slen, bs)
                positions = positions.transpose(0, 1)

        # do not recompute cached elements
        if use_cache and (not self.use_library_attention):
            _slen = slen - self.cache["slen"]
            x = x[:, -_slen:]
            # positions = positions[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]
        else:
            use_cache = False

        # all layer outputs
        if DataTransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs = []

        tensor = x

        if self.position_embeddings is not None:
            tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        if DataTransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs.append(tensor.detach().cpu())

        # transformer layers
        for i in range(self.n_layers):
            # self attention
            if use_cache:
                self.attentions[i].cache = self.cache
            attn = self.attentions[i](tensor, attn_mask, use_cache=use_cache)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # encoder attention (for decoder only)
            if self.is_decoder and src_enc is not None:
                if use_cache:
                    self.encoder_attn[i].cache = self.cache
                attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, use_cache=use_cache)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm15[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)

            tensor *= mask.unsqueeze(-1).to(tensor.dtype)
            if DataTransformerModel.STORE_OUTPUTS and not self.training:
                self.outputs.append(tensor.detach().cpu())

        # update cache length
        if use_cache:
            self.cache["slen"] += tensor.size(1)

        # move back sequence length to dimension 0
        tensor = tensor.transpose(0, 1)

        return tensor  # (slen, bs, dim)

    def predict(self, tensor, pred_mask, y, window_size):
        """
        Given the last hidden state, compute word scores and/or the loss.
        Inputs:
            tensor        (window_size+data_len, bs, dim)
            pred_mask     ByteTensor (slen, bs), filled with 1 when we need to predict a word
            y             LongTensor of shape (pred_mask.sum(),)
            get_scores    Boolean specifying whether we need to return scores
        """
        x = tensor[window_size - 1 : -1, :, :]
        # x = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)

        scores = self.proj(x).squeeze(dim=-1)
        loss = F.mse_loss(scores.float(), y, reduction="mean")
        return scores, loss

    def generate(self, src_enc, src_len, initial_values, t_pred, embedder, window_size):
        """
        Decode a sentence given initial start.
        Inputs:
            src_enc          (bs, slen, dim) fused features
            src_len          tuple of data and text lengths
            initial_values   (bs, window_size) initial values of the predicted sequences
            t_pred           (max_len, ) window_size + times at which to predict the values
        """

        # input batch
        max_len = t_pred.size(0)  # (input_num + window_size)
        data_len, text_len = src_len[0], src_len[1]  # (bs, )
        bs = data_len.size(0)
        assert src_enc.size(0) == bs

        # generated sentences
        generated = torch.zeros(max_len, bs, dtype=t_pred.dtype, device=t_pred.device)  # use 0 as data padding
        generated[0:window_size, :] = initial_values.transpose(0, 1)

        # current position / max lengths / length of generated sentences / unfinished sentences
        cur_len = window_size
        gen_len = text_len.clone().fill_(window_size)  # (bs, )
        # unfinished_sents = text_len.clone().fill_(1)

        # cache compute states
        self.cache = {"slen": 0}
        self.cache["embed_data_input"] = torch.zeros(max_len, bs, self.dim, dtype=t_pred.dtype, device=t_pred.device)
        while cur_len < max_len:
            if cur_len == window_size:
                embedder_input = torch.cat(
                    (
                        t_pred[:window_size].view(1, window_size, 1).expand(bs, window_size, 1),
                        initial_values.view(bs, window_size, 1),
                    ),
                    dim=2,
                )  # (bs, window_size, 2)
                cur_input = embedder(embedder_input)  # (bs, window_size, emb_dim)
                embed_data_input = self.cache["embed_data_input"]
                embed_data_input[:window_size, :, :] = cur_input.transpose(0, 1)
            else:
                idx = cur_len - 1
                embedder_input = torch.full((bs, 2), t_pred[idx], device=t_pred.device)
                embedder_input[:, 1] = generated[idx, :]
                cur_input = embedder(embedder_input)  # (bs, emb_dim)
                embed_data_input = self.cache["embed_data_input"]  # (max_len, bs, emb_dim)
                embed_data_input[idx, :, :] = cur_input

            # compute word scores
            tensor = self.forward(
                "fwd",
                x=embed_data_input[cur_len - window_size : cur_len, :, :],  # (window_size, bs, emb_dim)
                lengths=gen_len,
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                use_cache=True,
            )  # (cur_len, bs, dim)

            tensor = tensor[-1, :, :].to(self.dtype)  # (bs, dim)
            next_values = self.proj(tensor).flatten()  # (bs, )

            # update generations / lengths / finished sentences / current length
            generated[cur_len, :] = next_values
            # gen_len.add_(unfinished_sents)
            cur_len = cur_len + 1

        return generated  # (max_len, bs)


class DataOperatorModel(nn.Module):
    STORE_OUTPUTS = GLOBAL_STORE_OUTPUTS

    def __init__(
        self,
        params,
        with_output,
        positional_embeddings,
    ):
        """
        Transformer model (encoder or decoder) for data.
        """
        super().__init__()

        # encoder / decoder, output layer
        self.dtype = torch.float
        self.with_output = with_output

        # model parameters

        self.dim = params.data_dec_emb_dim  # 512 by default
        self.src_dim = params.data_enc_emb_dim
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_hidden_layers = params.n_data_dec_hidden_layers
        self.n_heads = params.n_data_dec_heads  # 8 by default
        self.n_layers = params.n_data_dec_layers

        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        self.norm_attention = params.norm_attention
        self.use_decoder_attn = params.data_decoder_attn
        assert self.dim % self.n_heads == 0, "transformer dim must be a multiple of n_heads"

        # embeddings

        self.query_embedder = nn.Linear(1, self.dim)

        if positional_embeddings is None or positional_embeddings == "alibi":
            self.position_embeddings = None
        elif positional_embeddings == "sinusoidal":
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
            create_sinusoidal_embeddings(N_MAX_POSITIONS, self.dim, out=self.position_embeddings.weight)
        elif positional_embeddings == "learnable":
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        else:
            raise NotImplementedError

        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)

        # transformer layers
        if params.data_feature_resnet:
            if not params.no_text and not params.split_fused_feature_data:
                self.text_embedder = nn.Sequential(
                    nn.Linear(self.dim, self.dim * 2),
                    nn.GELU(),
                    nn.Linear(self.dim * 2, self.dim),
                )
            self.data_embedder = nn.Sequential(
                nn.Linear(self.dim, self.dim * 2),
                nn.GELU(),
                nn.Linear(self.dim * 2, self.dim),
            )

        self.use_library_attention = params.use_library_attention
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        if self.use_decoder_attn:
            self.layer_norm15 = nn.ModuleList()
            self.encoder_attn = nn.ModuleList()

        for layer_id in range(self.n_layers):
            self.attentions.append(
                MultiHeadAttention(
                    self.n_heads,
                    self.dim,
                    self.dim,
                    dropout=self.attention_dropout,
                    normalized_attention=self.norm_attention,
                    use_library=params.use_library_attention,
                )
            )
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            if self.use_decoder_attn:
                self.layer_norm15.append(nn.LayerNorm(self.dim, eps=1e-12))
                self.encoder_attn.append(
                    MultiHeadAttention(
                        self.n_heads,
                        self.dim,
                        self.src_dim,
                        dropout=self.attention_dropout,
                        normalized_attention=self.norm_attention,
                        use_library=params.use_library_attention,
                    )
                )
            self.ffns.append(
                TransformerFFN(
                    self.dim,
                    self.hidden_dim,
                    self.dim,
                    self.n_hidden_layers,
                    dropout=self.dropout,
                )
            )
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))

        self.cache = None

        # output layer
        if self.with_output:
            self.proj = nn.Linear(self.dim, params.max_output_dimension, bias=True)
        self.split_fused_feature = params.split_fused_feature_data
        self.no_text = params.no_text
        self.data_feature_resnet = params.data_feature_resnet

    def get_query_emb(self, query_times):
        slen = query_times.size(0)
        query_times = query_times.view(slen, 1)
        query_emb = self.query_embedder(query_times)
        return query_emb  # (slen, dim)

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "query_emb":
            return self.get_query_emb(**kwargs)
        elif mode == "predict":
            return self.predict(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(
        self,
        query_emb,
        src_enc,
        src_len,
        positions=None,
        use_cache=False,
    ):
        """
        Inputs:
            query_emb   (slen, dim), embedding of times for evaluation
            src_enc     (bs, slen, dim), output from fused features
            src_len     LongTensor(bs), containing the length of src_enc
            positions   LongTensor(slen, bs), containing word positions
        """

        # check inputs
        bs = src_enc.size(0)
        slen, dim = query_emb.size(0), query_emb.size(1)
        query_emb = query_emb.view(1, slen, dim).expand(bs, slen, dim)

        if self.no_text:
            src_data_len = src_len[0]
            max_data_len = src_data_len.max()
            src_data_mask = (
                torch.arange(max_data_len, dtype=torch.long, device=query_emb.device) < src_data_len[:, None]
            )  # (bs, data_len)

            src_mask = src_data_mask
            if self.data_feature_resnet:
                src_enc = src_enc + self.data_embedder(src_enc)

        else:
            src_data_len, src_text_len = src_len[0], src_len[1]
            max_data_len = src_data_len.max()
            src_data_mask = (
                torch.arange(max_data_len, dtype=torch.long, device=query_emb.device) < src_data_len[:, None]
            )  # (bs, data_len)

            if self.split_fused_feature:
                src_mask = src_data_mask

                src_data = src_enc[:, :max_data_len, :]
                if self.data_feature_resnet:
                    src_data = src_data + self.data_embedder(src_data)
                src_enc = src_data
            else:
                src_text_mask = (
                    torch.arange(src_text_len.max(), dtype=torch.long, device=query_emb.device) < src_text_len[:, None]
                )  # (bs, text_len)

                src_mask = torch.cat([src_data_mask, src_text_mask], dim=1)
                if self.data_feature_resnet:
                    src_data = src_enc[:, :max_data_len, :]
                    src_text = src_enc[:, max_data_len:, :]
                    src_data = src_data + self.data_embedder(src_data)
                    src_text = src_text + self.text_embedder(src_text)
                    src_enc = torch.cat([src_data, src_text], dim=1)

        # positions
        if self.position_embeddings is not None:
            if positions is None:
                positions = query_emb.new(slen).long()
                positions = torch.arange(slen, out=positions).unsqueeze(0)
            else:
                assert positions.size() == (slen, bs)
                positions = positions.transpose(0, 1)

        # all layer outputs
        if DataOperatorModel.STORE_OUTPUTS and not self.training:
            self.outputs = []

        tensor = query_emb  # (bs, slen, dim)

        if self.position_embeddings is not None:
            tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        if DataOperatorModel.STORE_OUTPUTS and not self.training:
            self.outputs.append(tensor.detach().cpu())

        # transformer layers
        for i in range(self.n_layers):
            # self attention
            attn = self.attentions[i](tensor, src_mask, kv=src_enc)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # encoder attention (for decoder only)
            if self.use_decoder_attn:
                attn = self.encoder_attn[i](tensor, use_cache=use_cache)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm15[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)

            if DataOperatorModel.STORE_OUTPUTS and not self.training:
                self.outputs.append(tensor.detach().cpu())

        # move back sequence length to dimension 0
        tensor = tensor.transpose(0, 1)

        return tensor  # (slen, bs, dim)

    def predict(self, tensor, pred_mask, y, weight=None):
        """
        Given the last hidden state, compute word scores and/or the loss.
        Inputs:
            tensor     (slen, bs, dim)
            pred_mask  (slen, bs, output_dim) mask for different dimension/length
            y          (pred_mask.sum(), ) labels for prediction
            weight     (pred_mask.sum(), ) weight for loss function
        """
        scores = self.proj(tensor)  # (slen, bs, output_dim)
        scores = scores[pred_mask]
        loss = F.mse_loss(scores.float(), y, reduction="none")
        if weight is None:
            # no reweighting, loss is just regular MSE
            loss = torch.mean(loss)
        else:
            # reweight by weight
            loss = torch.sum(loss * weight)
        return scores, loss

    def generate(
        self,
        src_enc,
        src_len,
        query_emb,
    ):
        """
        Generate a sequence at times specified in query_emb
        Inputs:
            src_enc    (bs, slen, dim) fused features
            src_len    tuple of data and text lengths
            query_emb  (slen, dim)
        """

        tensor = self.forward(
            "fwd",
            query_emb=query_emb,
            src_enc=src_enc,
            src_len=src_len,
        )  # (slen, bs, dim)

        return self.proj(tensor).float()  # (slen, bs, output_dim)


class TextTransformerModel(nn.Module):
    STORE_OUTPUTS = GLOBAL_STORE_OUTPUTS

    def __init__(
        self,
        params,
        id2word,
        is_encoder,
        with_output,
        use_prior_embeddings,
        positional_embeddings,
    ):
        """
        Transformer model (encoder or decoder) for text.
        """
        super().__init__()

        # encoder / decoder, output layer
        self.dtype = torch.float
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output

        # dictionary

        self.id2word = id2word
        self.word2id = {s: i for i, s in self.id2word.items()}
        self.bos_index = self.word2id["<BOS>"]
        self.eos_index = self.word2id["<EOS>"]
        self.pad_index = self.word2id["<PAD>"]

        self.n_words = len(self.id2word)
        assert len(self.id2word) == self.n_words

        # model parameters

        self.dim = params.text_enc_emb_dim if is_encoder else params.text_dec_emb_dim  # 512 by default
        self.src_dim = params.text_enc_emb_dim
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_hidden_layers = params.n_text_enc_hidden_layers if is_encoder else params.n_text_dec_hidden_layers
        self.n_heads = params.n_text_enc_heads if is_encoder else params.n_text_dec_heads  # 8 by default
        self.n_layers = params.n_text_enc_layers if is_encoder else params.n_text_dec_layers

        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        self.norm_attention = params.norm_attention
        assert self.dim % self.n_heads == 0, "transformer dim must be a multiple of n_heads"

        # embeddings

        if positional_embeddings is None or positional_embeddings == "alibi":
            self.position_embeddings = None
        elif positional_embeddings == "sinusoidal":
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
            create_sinusoidal_embeddings(N_MAX_POSITIONS, self.dim, out=self.position_embeddings.weight)
        elif positional_embeddings == "learnable":
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        else:
            raise NotImplementedError

        self.use_prior_embeddings = use_prior_embeddings
        if not use_prior_embeddings:
            self.embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        else:
            self.embeddings = None
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)

        # transformer layers
        self.use_library_attention = params.use_library_attention
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        if self.is_decoder:
            self.layer_norm15 = nn.ModuleList()
            self.encoder_attn = nn.ModuleList()

        for layer_id in range(self.n_layers):
            self.attentions.append(
                MultiHeadAttention(
                    self.n_heads,
                    self.dim,
                    self.dim,
                    dropout=self.attention_dropout,
                    normalized_attention=self.norm_attention,
                    use_library=params.use_library_attention,
                )
            )
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            if self.is_decoder:
                self.layer_norm15.append(nn.LayerNorm(self.dim, eps=1e-12))
                self.encoder_attn.append(
                    MultiHeadAttention(
                        self.n_heads,
                        self.dim,
                        self.src_dim,
                        dropout=self.attention_dropout,
                        normalized_attention=self.norm_attention,
                        use_library=params.use_library_attention,
                    )
                )
            self.ffns.append(
                TransformerFFN(
                    self.dim,
                    self.hidden_dim,
                    self.dim,
                    self.n_hidden_layers,
                    dropout=self.dropout,
                )
            )
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))

        self.cache = None
        self.split_fused_feature = params.split_fused_feature_text

        # output layer
        if self.with_output:
            assert not self.use_prior_embeddings
            self.proj = nn.Linear(self.dim, self.n_words, bias=True)  ##added index for eos and tab
            if params.text_share_inout_emb:
                self.proj.weight = self.embeddings.weight

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "predict":
            return self.predict(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(
        self,
        x,
        lengths,
        causal,
        src_enc=None,
        src_len=None,
        positions=None,
        use_cache=False,
    ):
        """
        Inputs:
            x          LongTensor(slen, bs), containing word indices
            lengths    LongTensor(bs), containing the length of each sentence
            causal     Boolean, if True, the attention is only done over previous hidden states
            positions  LongTensor(slen, bs), containing word positions
        """

        # check inputs
        slen, bs = x.size()[:2]
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)  # batch size as dimension 0
        assert (src_enc is None) == (src_len is None)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.size(0) == bs
        assert not (use_cache and self.cache is None)

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, causal)
        if self.is_decoder and src_enc is not None:
            if use_cache and ("src_mask" in self.cache):
                src_mask = self.cache["src_mask"]
                if self.split_fused_feature:
                    max_data_len = self.cache["max_data_len"]
                    src_enc = src_enc[:, max_data_len:, :]
            else:
                src_data_len, src_text_len = src_len[0], src_len[1]
                max_data_len = src_data_len.max()

                src_text_mask = (
                    torch.arange(src_text_len.max(), dtype=torch.long, device=lengths.device) < src_text_len[:, None]
                )  # (bs, text_len)

                if self.split_fused_feature:
                    src_mask = src_text_mask
                    src_enc = src_enc[:, max_data_len:, :]
                else:
                    src_data_mask = (
                        torch.arange(max_data_len, dtype=torch.long, device=lengths.device) < src_data_len[:, None]
                    )  # (bs, data_len)
                    src_mask = torch.cat([src_data_mask, src_text_mask], dim=1)
                if use_cache:
                    self.cache["src_mask"] = src_mask
                    self.cache["max_data_len"] = max_data_len

        # positions
        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)

        # do not recompute cached elements
        if use_cache and (not self.use_library_attention):
            _slen = slen - self.cache["slen"]
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]
        else:
            use_cache = False

        # all layer outputs
        if TextTransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs = []

        # embeddings
        if not self.use_prior_embeddings:
            tensor = self.embeddings(x)
        else:
            tensor = x

        if self.position_embeddings is not None:
            tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        if TextTransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs.append(tensor.detach().cpu())

        # transformer layers
        for i in range(self.n_layers):
            # self attention
            if use_cache:
                self.attentions[i].cache = self.cache
            attn = self.attentions[i](tensor, attn_mask, use_cache=use_cache)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # encoder attention (for decoder only)
            if self.is_decoder and src_enc is not None:
                if use_cache:
                    self.encoder_attn[i].cache = self.cache
                attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, use_cache=use_cache)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm15[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)

            tensor *= mask.unsqueeze(-1).to(tensor.dtype)
            if TextTransformerModel.STORE_OUTPUTS and not self.training:
                self.outputs.append(tensor.detach().cpu())

        # update cache length
        if use_cache:
            self.cache["slen"] += tensor.size(1)

        # move back sequence length to dimension 0
        tensor = tensor.transpose(0, 1)

        return tensor  # (slen, bs, dim)

    def predict(self, tensor, pred_mask, y, get_scores):
        """
        Given the last hidden state, compute word scores and/or the loss.
        Inputs:
            pred_mask    ByteTensor (slen, bs), filled with 1 when we need to predict a word
            y            LongTensor (pred_mask.sum(),)
            get_scores   Boolean specifying whether we need to return scores
        """
        x = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        assert (y == self.pad_index).sum().item() == 0
        scores = self.proj(x).view(-1, self.n_words)
        loss = F.cross_entropy(scores.float(), y, reduction="mean")
        return scores, loss

    def generate(self, src_enc, src_len, max_len=200, top_p=1.0, sample_temperature=None):
        """
        Decode a sentence given initial start.
        Inputs:
            src_end     (bs, slen, dim) fused features
            src_len     tuple of data and text lengths
        Outputs:
            x           LongTensor(bs, slen)
                            <BOS> W1 W2 W3 <EOS> <PAD>
                            <BOS> W1 W2 W3   W4  <EOS>
            lengths     LongTensor(bs)
                            [5, 6]
        """

        # input batch
        data_len, text_len = src_len[0], src_len[1]  # (bs, )
        bs = data_len.size(0)
        assert src_enc.size(0) == bs

        # generated sentences
        generated = text_len.new(max_len, bs)  # upcoming output
        generated.fill_(self.pad_index)  # fill upcoming ouput with <PAD>
        generated[0].fill_(self.bos_index)  # we use <EOS> for <BOS> everywhere

        # positions
        positions = text_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)

        # current position / max lengths / length of generated sentences / unfinished sentences
        cur_len = 1
        gen_len = text_len.clone().fill_(1)  # (bs, )
        unfinished_sents = text_len.clone().fill_(1)

        # cache compute states
        self.cache = {"slen": 0}
        while cur_len < max_len:
            # compute word scores
            tensor = self.forward(
                "fwd",
                x=generated[:cur_len],  # (cur_len, bs)
                lengths=gen_len,
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                use_cache=True,
            )
            # assert tensor.size() == (cur_len, bs, self.dim)

            tensor = tensor.data[-1, :, :].to(self.dtype)  # (bs, dim)  ##BE CAREFUL
            scores = self.proj(tensor)  # (bs, n_words)

            # select next words: sample or greedy
            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(
                    F.softmax(scores.float() / sample_temperature, dim=1), num_samples=1
                ).squeeze(1)
            assert next_words.size() == (bs,)

            # update generations / lengths / finished sentences / current length
            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (1 - unfinished_sents)
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # add <EOS> to unfinished sentences
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.bool(), self.eos_index)
        # sanity check
        # assert (generated == self.eos_index).sum() == 2 * bs
        generated = generated.unsqueeze(-1).view(generated.shape[0], bs)
        return generated[:cur_len], gen_len

    def generate_beam(
        self,
        src_enc,
        src_len,
        beam_size,
        length_penalty,
        early_stopping,
        max_len=200,
    ):
        """
        Decode a sentence given initial start.
        Outputs:
            x           LongTensor(bs, slen)
                            <EOS> W1 W2 W3 <EOS> <PAD>
                            <EOS> W1 W2 W3   W4  <EOS>
            lengths     LongTensor(bs)
                            [5, 6]
            positions   - False, for regular "arange" positions (LM)
                        - True, to reset positions from the new generation (MT)
        """

        # check inputs
        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1
        # batch size / number of words
        bs = len(src_len)
        n_words = self.n_words

        # expand to beam size the source latent representations / source lengths
        src_enc = (
            src_enc.unsqueeze(1)
            .expand((bs, beam_size) + src_enc.shape[1:])
            .contiguous()
            .view((bs * beam_size,) + src_enc.shape[1:])
        )
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

        # generated sentences (batch with beam current hypotheses)
        generated = src_len.new(max_len, bs * beam_size)  # upcoming output
        generated.fill_(self.pad_index)  # fill upcoming ouput with <PAD>
        generated[0].fill_(self.bos_index)  # we use <EOS> for <BOS> everywhere

        # generated hypotheses
        generated_hyps = [BeamHypotheses(beam_size, max_len, length_penalty, early_stopping) for _ in range(bs)]

        # positions
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)

        # scores for each sentence in the beam
        beam_scores = src_enc.new(bs, beam_size).float().fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # current position
        cur_len = 1

        # cache compute states
        self.cache = {"slen": 0}

        # done sentences
        done = [False for _ in range(bs)]

        while cur_len < max_len:
            # compute word scores
            tensor = self.forward(
                "fwd",
                x=generated[:cur_len],
                lengths=src_len.new(bs * beam_size).fill_(cur_len),
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                use_cache=True,
            )

            assert tensor.size() == (1, bs * beam_size, self.dim)
            tensor = tensor.data[-1, :, :]  # (bs * beam_size, dim)
            scores = self.proj(tensor)  # (bs * beam_size, n_words)
            scores = F.log_softmax(scores.float(), dim=-1)  # (bs * beam_size, n_words)
            assert scores.size() == (bs * beam_size, n_words)

            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
            _scores = _scores.view(bs, beam_size * n_words)  # (bs, beam_size * n_words)

            next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for sent_id in range(bs):
                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, self.pad_index, 0)] * beam_size)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):
                    # get beam and word IDs
                    beam_id = torch.div(idx, n_words, rounding_mode="trunc")
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(
                            generated[:cur_len, sent_id * beam_size + beam_id].clone().cpu(),
                            value.item(),
                        )
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.pad_index, 0)] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            # re-order batch and internal states
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in self.cache.keys():
                if k != "slen":
                    self.cache[k] = (
                        self.cache[k][0][beam_idx],
                        self.cache[k][1][beam_idx],
                    )
            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # def get_coeffs(s):
        #     roots = [int(s[i + 2]) for i, c in enumerate(s) if c == 'x']
        #     poly = np.poly1d(roots, r=True)
        #     coeffs = list(poly.coefficients.astype(np.int64))
        #     return [c % 10 for c in coeffs], coeffs

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(bs):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         hh = " ".join(self.id2word[x] for x in ww.tolist())
        #         print(f"{ss:+.4f} {hh}")
        #         # cc = get_coeffs(hh[4:])
        #         # print(f"{ss:+.4f} {hh} || {cc[0]} || {cc[1]}")
        #     print("")

        # select the best hypotheses
        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = src_len.new(tgt_len.max().item(), bs).fill_(self.pad_index)
        for i, hypo in enumerate(best):
            decoded[: tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = self.eos_index

        # sanity check
        # assert (decoded == self.eos_index).sum() == 2 * bs

        return decoded, tgt_len, generated_hyps


class BeamHypotheses(object):
    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1  # ignoring <BOS>
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap,
        then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_len**self.length_penalty


def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        logits = TopKLogitsWarper(
            top_k=top_k,
            filter_value=filter_value,
            min_tokens_to_keep=min_tokens_to_keep,
        )(None, logits)

    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=min_tokens_to_keep)(None, logits)

    return logits


class LogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Torch method for warping logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class TopKLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.
    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(
        self,
        top_k: int,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.
    Args:
        top_p (`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to `top_p` or higher are kept
            for generation.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(
        self,
        top_p: float,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores
