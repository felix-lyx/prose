from logging import getLogger
import torch
from tabulate import tabulate

from .transformer_wrappers import PROSE_1to1, PROSE_2to1
from .baselines import FNO, UNet, ViT, DeepONet

logger = getLogger()


def build_model(params, model_config, data_config, symbol_env):

    modules = {}

    # get model
    name = model_config.name

    if name == "prose_1to1":
        modules["model"] = PROSE_1to1(
            model_config,
            symbol_env,
            data_config.x_num,
            data_config.max_output_dimension,
            data_config.t_num - params.input_len,
        )

    elif name == "prose_2to1":
        modules["model"] = PROSE_2to1(
            model_config,
            symbol_env,
            data_config.x_num,
            data_config.max_output_dimension,
            data_config.t_num - params.input_len,
        )

    elif name == "fno":
        modules["model"] = FNO(model_config, data_config.max_output_dimension)

    elif name == "vit":
        modules["model"] = ViT(model_config, data_config.x_num, data_config.max_output_dimension)

    elif name == "unet":
        modules["model"] = UNet(model_config, data_config.max_output_dimension)

    elif name == "deeponet":
        modules["model"] = DeepONet(model_config, data_config, params.input_len)

    else:
        assert False, f"Model {name} hasn't been implemented"

    # reload pretrained modules
    if params.reload_model:
        logger.info(f"Reloading modules from {params.reload_model} ...")
        reloaded = torch.load(params.reload_model)
        for k, v in modules.items():
            assert k in reloaded, f"{k} not in save"
            if all([k2.startswith("module.") for k2 in reloaded[k].keys()]):
                reloaded[k] = {k2[len("module.") :]: v2 for k2, v2 in reloaded[k].items()}
            if all([k2.startswith("_orig_mod.") for k2 in reloaded[k].keys()]):
                reloaded[k] = {k2[len("_orig_mod.") :]: v2 for k2, v2 in reloaded[k].items()}
            v.load_state_dict(reloaded[k])

    # log
    for k, v in modules.items():
        logger.info(f"{k}: {v}")
    for k, v in modules.items():
        s = f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad]):,}"
        if hasattr(v, "summary"):
            # for individual components of a wrapper model
            s += v.summary()
        logger.info(s)

    # for k, v in modules.items():
    #     table_data = [(name, str(param.shape), param.requires_grad) for name, param in v.named_parameters()]
    #     logger.info("\n" + tabulate(table_data, headers=["Parameter Name", "Shape", "Requires Grad"], tablefmt="grid"))
    #     table_data = [(name, str(param.shape)) for name, param in v.named_parameters() if param.requires_grad]
    #     logger.info("\n" + tabulate(table_data, headers=["Trainable Parameters", "Shape"], tablefmt="grid"))

    # cuda
    if not params.cpu:
        for v in modules.values():
            v.cuda()

    if params.compile:
        for k, v in modules.items():
            # modules[k] = torch.compile(v, mode="reduce-overhead")
            modules[k] = torch.compile(v)

    return modules
