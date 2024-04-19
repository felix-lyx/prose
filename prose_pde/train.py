import torch.multiprocessing as multiprocessing

try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass  # The context has already been set, so you can ignore the error.

import random
import numpy as np
import torch
import os
import h5py

os.environ["JAX_PLATFORM_NAME"] = "cpu"

# import jax
# print("Default JAX backend:", jax.default_backend())

from pathlib import Path

import symbolicregression
from symbolicregression.mode import init_distributed_mode
from symbolicregression.utils import initialize_exp
from symbolicregression.model import check_model_params, build_modules
from symbolicregression.envs import build_env
from symbolicregression.trainer import Trainer
from evaluate import Evaluator
from parsers import get_parser
from omegaconf import DictConfig
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

import wandb

# np.seterr(all="raise")
np.seterr(divide="raise", under="ignore", over="raise", invalid="raise")


def main(params):
    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)
    logger = initialize_exp(params)

    # CPU / CUDA
    if not params.cpu:
        assert torch.cuda.is_available()
    symbolicregression.utils.CUDA = not params.cpu

    # wandb logging

    if not params.is_master:
        params.use_wandb = False
    if params.use_wandb:
        wandb_id = wandb.util.generate_id() if params.wandb_id is None else params.wandb_id
        params.wandb_id = wandb_id
        wandb.init(
            project=params.exp_name,
            resume="allow",
            id=wandb_id,
            name=params.wandb_name,
        )
        wandb.config.update(params, allow_val_change=True)
        wandb.log({"id": wandb_id})

    # build environment / modules / trainer / evaluator
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)
    if params.eval_dump_path is None:
        params.eval_dump_path = Path(params.dump_path) / "evals_all"
        if not os.path.isdir(params.eval_dump_path):
            os.makedirs(params.eval_dump_path)

    env = build_env(params)
    try:
        params.x_grid_size = env.generator.x_grid_size
    except:
        params.x_grid_size = 1
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    evaluator = Evaluator(trainer)

    if params.reload_data != "":
        data_types = ["valid{}".format(i) for i in range(1, len(trainer.data_path["functions"]))]
    else:
        data_types = ["valid1"]
    evaluator.set_env_copies(data_types)

    # evaluation
    if params.eval_only:
        if params.eval_in_domain:
            stats = evaluator.evaluate_in_domain(
                "valid1",
                "functions",
                save=params.save_results,
            )
            logger.info(
                "Epoch {} Eval | Data loss = {:.8f} | Abs Data loss = {:.8f} | First Half = {:.8f} | Second Half = {:.8f} | Mean Data loss = {:.8f} | L1 loss = {:.8f} | R2 loss = {:.8f} | Valid Fraction {} | Text loss = {:.8f}".format(
                    trainer.epoch,
                    stats["data_loss"],
                    stats["data_loss_abs"],
                    stats["data_loss_first_half"],
                    stats["data_loss_second_half"],
                    stats["data_loss_mean"],
                    stats["l1_loss"],
                    stats["r2_loss"],
                    stats["valid_fraction"],
                    stats["text_loss"],
                )
            )
            if params.use_wandb:
                stats["epoch"] = trainer.epoch
                wandb.log({"val": stats})

        exit()
     #compute losses
    # if params.compute_losses_only:
    #     if params.eval_in_domain:
    #         file = (
    #                  params.eval_dump_path if params.eval_dump_path is not None else params.dump_path
    #                 )
    #         file = os.path.join(file, "eval_in_domain.h5")
    #
    #         assert os.path.exists(file)
    #
    #     exit()

    trainer.n_equations = 0
    for _ in range(params.max_epoch):
        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.inner_epoch = 0
        while trainer.inner_epoch < trainer.n_steps_per_epoch:
            # training steps
            for task_id in np.random.permutation(len(params.tasks)):
                task = params.tasks[task_id]
                if params.export_data:
                    trainer.export_data(task)
                    if (params.log_periodic > 0) and (trainer.inner_epoch % params.log_periodic == 0):
                        logger.info("Epoch {} | Step {}".format(trainer.epoch, trainer.inner_epoch))
                else:
                    trainer.enc_dec_step(task)

                    if (params.log_periodic > 0) and (trainer.inner_epoch % params.log_periodic == 0):
                        data_loss = trainer.data_loss / params.log_periodic
                        text_loss = trainer.text_loss / params.log_periodic
                        logger.info(
                            "Epoch {} | Step {} | Train Data loss = {:.8f} | Train Text loss = {:.8f}".format(
                                trainer.epoch, trainer.inner_epoch, data_loss, text_loss
                            )
                        )
                        if params.use_wandb:
                            wandb.log(
                                {
                                    "train": {
                                        "data_loss": data_loss,
                                        "text_loss": text_loss,
                                        "epoch": trainer.epoch,
                                        "step": trainer.n_total_iter,
                                    }
                                }
                            )

                        trainer.data_loss = 0.0
                        trainer.text_loss = 0.0
                trainer.iter()

        logger.info("============ End of epoch %i ============" % trainer.epoch)

        trainer.save_periodic()

        if params.eval_in_domain:
            stats = evaluator.evaluate_in_domain(
                "valid1",
                "functions",
                save=params.save_results,
            )
            logger.info("Epoch {} Eval | Data loss = {:.8f} | Abs Data loss = {:.8f} | First Half = {:.8f} | Second Half = {:.8f} | Mean Data loss = {:.8f} | L1 loss = {:.8f} | R2 loss = {:.8f} | Valid Fraction {} | Text loss = {:.8f}".format(
             trainer.epoch,
             stats["data_loss"],
             stats["data_loss_abs"],
             stats["data_loss_first_half"],
             stats["data_loss_second_half"],
             stats["data_loss_mean"],
             stats["l1_loss"],
             stats["r2_loss"],
             stats["valid_fraction"],
             stats["text_loss"],
         )
                         )
            if params.use_wandb:
                stats["epoch"] = trainer.epoch
                wandb.log({"val": stats})

            if params.text_only:
                if stats["text_loss"] >= 0.00001:
                    trainer.save_best_model({"_text_loss": stats["text_loss"]})
            else:
                trainer.save_best_model(
                    {
                        "_data_loss": stats["data_loss"],
                        "_text_loss": stats["text_loss"],
                        "_total_loss": stats["total_loss"],
                    }
                )

        # end of epoch
        trainer.end_epoch()

    if params.export_data and params.separate_modality:
        with h5py.File(params.export_path_data, "w") as hf:
            hf.create_dataset("data", data=trainer.data_matrix)
        logger.info(f"Data have been stored in h5 in: {params.export_path_data}.")

    max_mem = torch.cuda.max_memory_allocated() / 1024**2
    s_mem = " MEM: {:.2f} MB ".format(max_mem)
    logger.info(s_mem)

    if params.multi_gpu:
        torch.distributed.destroy_process_group()

    if params.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    if params.dry_run:
        print("Debugging run...")
        params.max_epoch = 1
        params.n_steps_per_epoch = 20
        params.debugging = True
        params.exp_name = "debugging"
        params.use_wandb = False
        params.save_periodic = 0
        params.log_periodic = 10
        params.batch_size_eval = 32
        params.eval_size = params.batch_size_eval
    else:
        params.debugging = False

    if params.export_data:
        params.cpu = True
        params.eval_in_domain = False
        params.use_wandb = False
        params.save_periodic = 0
        params.amp = -1
        params.reload_data = ""
        params.text_enc_emb_dim = 64
        params.text_dec_emb_dim = 64
        params.data_enc_emb_dim = 64
        params.data_dec_emb_dim = 64
        params.fusion_emb_dim = 64

    if params.text_only:
        params.validation_metrics = "_text_loss"

    if params.eval_only and params.eval_from_exp != "":
        if os.path.exists(params.eval_from_exp + "/best-" + params.validation_metrics + ".pth"):
            params.reload_model = params.eval_from_exp + "/best-" + params.validation_metrics + ".pth"
        elif os.path.exists(params.eval_from_exp + "/checkpoint.pth"):
            params.reload_model = params.eval_from_exp + "/checkpoint.pth"
        else:
            assert os.path.exists(params.eval_from_exp)
            params.reload_model = params.eval_from_exp

        eval_data = params.eval_data

        if params.reload_data or params.eval_data:
            params.reload_data = params.tasks + "," + eval_data + "," + eval_data + "," + eval_data
        params.local_rank = -1
        params.master_port = -1

    # debug mode
    if params.debug:
        params.exp_name = "debug"
        if params.exp_id == "":
            params.exp_id = "debug_%08i" % random.randint(0, 100000000)

    # if params.types.startswith("pde"):
    #     params.max_output_dimension = 2 + params.max_pde_spatialdim* 2 + params.max_pde_spatialdim * (params.max_pde_spatialdim - 1) // 2
    # check parameters
    check_model_params(params)

    # run experiment
    main(params)
