from logging import getLogger
import os
import torch
import torch.nn as nn

from .transformer import TextTransformerModel, DataTransformerModel, DataOperatorModel, FusionTransformerModel, RevIN

logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # model dimensions
    assert params.text_enc_emb_dim % params.n_text_enc_heads == 0
    assert params.text_dec_emb_dim % params.n_text_dec_heads == 0
    assert params.data_enc_emb_dim % params.n_data_enc_heads == 0
    assert params.data_dec_emb_dim % params.n_data_dec_heads == 0
    assert params.fusion_emb_dim % params.n_fusion_heads == 0

    # reload a pretrained model
    if params.reload_model != "":
        print("Reloading model from ", params.reload_model)
        assert os.path.isfile(params.reload_model)


def build_modules(env, params):
    """
    Build modules.
    """

    modules = {}

    modules["embedder"] = nn.Sequential(
        # nn.Linear(1 + params.max_input_dimension + params.max_output_dimension, params.data_enc_emb_dim),
        nn.Linear(1 + params.max_output_dimension * params.x_patch_size, params.data_enc_emb_dim),
        nn.GELU(),
        nn.Linear(params.data_enc_emb_dim, params.data_enc_emb_dim),
    )

    if not params.no_text:
        modules["text_encoder"] = TextTransformerModel(
            params,
            env.equation_id2word,
            is_encoder=True,
            with_output=False,
            use_prior_embeddings=False,
            positional_embeddings=params.text_enc_positional_embeddings,
        )

    if not params.no_text and not params.data_only:
        modules["text_decoder"] = TextTransformerModel(
            params,
            env.equation_id2word,
            is_encoder=False,
            with_output=True,
            use_prior_embeddings=False,
            positional_embeddings=params.text_dec_positional_embeddings,
        )

    modules["data_encoder"] = DataTransformerModel(
        params,
        is_encoder=True,
        with_output=False,
        positional_embeddings=params.data_enc_positional_embeddings,
    )
    if params.normalization:
        modules["normalizer"] = RevIN(
            params,
        )

    if not params.text_only:
        modules["data_decoder"] = DataOperatorModel(
            params,
            with_output=True,
            positional_embeddings=params.data_dec_positional_embeddings,
        )

    if not params.no_text:
        modules["fusion"] = FusionTransformerModel(
            params,
            positional_embeddings=params.fusion_positional_embeddings,
        )

    # reload pretrained modules
    if params.reload_model != "":
        logger.info(f"Reloading modules from {params.reload_model} ...")
        reloaded = torch.load(params.reload_model)
        for k, v in modules.items():
            assert k in reloaded, f"{k} not in save"
            if all([k2.startswith("module.") for k2 in reloaded[k].keys()]):
                reloaded[k] = {k2[len("module.") :]: v2 for k2, v2 in reloaded[k].items()}
            v.load_state_dict(reloaded[k])

    # log
    for k, v in modules.items():
        logger.debug(f"{v}: {v}")
    for k, v in modules.items():
        logger.info(f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad])}")

    if params.compile:
        assert False, "torch.compile seems buggy"
        for k, v in modules.items():
            modules[k] = torch.compile(v, mode="reduce-overhead")

    # cuda
    if not params.cpu:
        for v in modules.values():
            v.cuda()

    return modules
