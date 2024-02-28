from logging import getLogger
import os
import torch
import torch.distributed as dist
import numpy as np
from copy import deepcopy

import symbolicregression
from symbolicregression.model.model_wrapper import ModelWrapper
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from tqdm import tqdm

# np.seterr(all="raise")
np.seterr(divide="raise", under="ignore", over="raise", invalid="raise")

logger = getLogger()


def ODE_solver(tree, IC, output_grid, logger=None, type=None):
    # solve ODE where RHS is given by tree

    y0 = IC.numpy(force=True)  # (dim,)
    t_span = [0.0, 6.0]

    def f(t, y):
        shape = y.shape  # (dim,)
        return tree.val(y.reshape(1, -1)).reshape(shape)

    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method="BDF",
            t_eval=output_grid,
            rtol=1e-4,
            atol=1e-6,
        )

        if (sol.status) == 0:
            return torch.from_numpy(sol.y.transpose().astype(np.single))  # (t_num, dim)
        else:
            logger.info(f"{type} solver error: sol status {sol.status}")
    except Exception as e:
        logger.info(f"{type} error is {e}")

    return None


def compute_losses(output, target, output_len, eps):
    """
    output: (output, dim)
    target: (output, dim)

    RETURN: MSE, relative SE, first half relative SE, second half relative SE
    """
    half_len = output_len // 2
    target_first_half_sum = torch.sum(target[:half_len] ** 2)
    target_second_half_sum = torch.sum(target[half_len:] ** 2)
    first_half_diff = torch.sum((output[:half_len] - target[:half_len]) ** 2)
    second_half_diff = torch.sum((output[half_len:] - target[half_len:]) ** 2)

    abs_loss = first_half_diff + second_half_diff
    rel_loss = torch.sqrt(abs_loss / (eps + target_first_half_sum + target_second_half_sum))
    rel_loss_first_half = torch.sqrt(first_half_diff / (eps + target_first_half_sum))
    rel_loss_second_half = torch.sqrt(second_half_diff / (eps + target_second_half_sum))

    return abs_loss / output_len / output.size(-1), rel_loss, rel_loss_first_half, rel_loss_second_half


class Evaluator(object):
    ENV = None

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        Evaluator.ENV = trainer.env
        self.eval_num_input_points = 50
        self.dataloader = None
        self.output_dim = self.params.max_output_dimension

        self.types = self.env.generator.ode_generator.types

        self.types_to_idx = {s: i for i, s in enumerate(self.types)}

    def set_env_copies(self, data_types):
        for data_type in data_types:
            setattr(self, "{}_env".format(data_type), deepcopy(self.env))

    def evaluate_in_domain(
        self,
        data_type,
        task,
        save=False,
        save_file=None,
    ):
        """
        Encoding / decoding step on the given evaluation dataset
        """

        params = self.params
        logger.info("====== STARTING EVALUATION (multi-gpu: {}) =======".format(params.multi_gpu))

        if "embedder" in self.modules:
            embedder = self.modules["embedder"].module if params.multi_gpu else self.modules["embedder"]
            embedder.eval()
        else:
            embedder = None

        if "text_encoder" in self.modules:
            text_encoder = self.modules["text_encoder"].module if params.multi_gpu else self.modules["text_encoder"]
            text_encoder.eval()
        else:
            text_encoder = None

        if "text_decoder" in self.modules:
            text_decoder = self.modules["text_decoder"].module if params.multi_gpu else self.modules["text_decoder"]
            text_decoder.eval()
        else:
            text_decoder = None

        if "data_encoder" in self.modules:
            data_encoder = self.modules["data_encoder"].module if params.multi_gpu else self.modules["data_encoder"]
            data_encoder.eval()
        else:
            data_encoder = None

        if "data_decoder" in self.modules:
            data_decoder = self.modules["data_decoder"].module if params.multi_gpu else self.modules["data_decoder"]
            data_decoder.eval()
        else:
            data_decoder = None

        if "fusion" in self.modules:
            fusion = self.modules["fusion"].module if params.multi_gpu else self.modules["fusion"]
            fusion.eval()
        else:
            fusion = None

        env = getattr(self, "{}_env".format(data_type))

        if self.params.eval_noise_gamma > 0:
            seed = [self.params.global_rank, self.params.test_env_seed]
            noise_rng = np.random.RandomState(seed)
            gamma = self.params.eval_noise_gamma

        eval_size_per_gpu = params.eval_size // params.n_gpu_per_node
        if self.dataloader is None:
            self.dataloader = env.create_test_iterator(
                data_type,
                task,
                data_path=self.trainer.data_path,
                batch_size=params.batch_size_eval,
                params=params,
                size=eval_size_per_gpu,
                test_env_seed=self.params.test_env_seed,
            )
        iterator = self.dataloader
        colors = ["blue", "orange", "green", "purple", "olive", "red", "magenta", "black"]

        mw = ModelWrapper(
            env=env,
            embedder=embedder,
            text_encoder=text_encoder,
            text_decoder=text_decoder,
            data_encoder=data_encoder,
            data_decoder=data_decoder,
            fusion=fusion,
            beam_length_penalty=params.beam_length_penalty,
            beam_size=params.beam_size,
            max_generated_output_len=params.max_generated_output_len,
            beam_early_stopping=params.beam_early_stopping,
            beam_temperature=params.beam_temperature,
            beam_type=params.beam_type,
            text_only=params.text_only,
            data_only=params.data_only,
            no_text=params.no_text,
            output_dim=self.output_dim,
            use_skeleton=params.use_skeleton,
            input_len=params.input_len,
            input_step=params.input_step,
            amp=params.eval_amp,
        )

        total_loss = torch.zeros(len(self.types), dtype=torch.float32)
        total_abs_loss = torch.zeros(len(self.types), dtype=torch.float32)
        total_count = torch.zeros(len(self.types), dtype=torch.long)

        if save:
            if save_file is None:
                save_file = (
                    self.params.eval_dump_path if self.params.eval_dump_path is not None else self.params.dump_path
                )
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_file = os.path.join(save_file, "eval_in_domain.csv")

        if not self.params.use_wandb:
            pbar = tqdm(total=eval_size_per_gpu)

        input_len = self.params.input_len
        t_eval = self.trainer.t_eval

        data_loss = []
        text_loss = 0.0
        text_valid = 0
        text_total = 0
        eps = 1e-6

        abs_data_loss = []
        data_loss_first_half = []
        data_loss_second_half = []

        if params.text_ode_solve:
            # use text output + ODE solver
            data_loss_valid = 0.0
            text_data_loss = 0.0
            text_valid_output = 0
            output_grid = env.generator.t_eval[input_len:]

        for samples, _ in iterator:
            data_seqs = samples["data"]  # (bs, data_len, output_dim)

            if self.params.eval_noise_gamma > 0:
                # add noise on data
                if self.params.noise_type == "multiplicative":
                    for i, seq in enumerate(data_seqs):
                        data_seqs[i] = seq + (
                            gamma
                            * torch.abs(seq)
                            * torch.from_numpy(noise_rng.randn(seq.size(0), seq.size(1)).astype(np.single))
                        )
                else:  # additive
                    for i, seq in enumerate(data_seqs):
                        cur_noise = torch.from_numpy(noise_rng.randn(seq.size(0), seq.size(1)).astype(np.single))
                        sigma = gamma * torch.linalg.vector_norm(seq) / (torch.linalg.vector_norm(cur_noise) + eps)
                        data_seqs[i] = seq + sigma * cur_noise

            if params.use_skeleton:
                text_seqs = samples["tree_skeleton"]  # (bs, text_len)
            else:
                text_seqs = samples["tree_encoded"]  # (bs, text_len)

            trees = samples["tree"]  # (bs, )
            bs = len(data_seqs)
            text_total += bs

            data_input = [seq[: input_len : params.input_step, :] for seq in data_seqs]
            data_outputs, text_outputs = mw(
                data_input=data_input,  # (bs, data_len, output_dim)
                text_input=text_seqs,  # (bs, text_len)
                logger=logger,
            )
            # data_outputs: torch tensor of shape (bs, data_len, output_dim)
            # text_outputs: nested list of shape (bs, tree_nums), some inner lists are possibly empty, elements are trees

            # computing eval losses

            input_points = np.random.uniform(-5, 5, size=(self.eval_num_input_points, self.output_dim))
            data_outputs = data_outputs.to(data_seqs[0].device)

            for i in range(bs):
                rel_loss = 0.0
                if not params.text_only:
                    # data loss
                    dim = data_seqs[i].size(-1)
                    (
                        abs_loss,
                        rel_loss,
                        rel_loss_first_half,
                        rel_loss_second_half,
                    ) = compute_losses(
                        data_outputs[i, :, :dim],
                        data_seqs[i][input_len:, :],
                        params.t_num - input_len,
                        eps,
                    )

                    abs_data_loss.append(abs_loss)
                    data_loss.append(rel_loss)
                    data_loss_first_half.append(rel_loss_first_half)
                    data_loss_second_half.append(rel_loss_second_half)

                    cur_type = samples["type"][i]
                    total_loss[self.types_to_idx[cur_type]] += rel_loss
                    total_abs_loss[self.types_to_idx[cur_type]] += abs_loss
                    total_count[self.types_to_idx[cur_type]] += 1

                    if params.print_outputs:
                        data = data_seqs[i]
                        output = data_outputs[i]
                        fig = plt.figure()
                        for j in range(dim):
                            c = colors[j]
                            plt.plot(t_eval, data[:, j], "--", linewidth=1.4, alpha=0.8, color=c, label=f"target {j}")
                            plt.plot(t_eval[input_len:], output[:, j], "-", linewidth=2, color=c, label=f"output {j}")
                        plt.legend(loc="best")
                        plt.title("{} | {} | {:.6f}".format(i, cur_type, rel_loss))
                        plt.savefig("figures/eval_{}.png".format(i))
                        plt.close(fig)

                if not params.data_only and not params.no_text:
                    # text loss
                    tree_list = text_outputs[i]
                    label_outputs = None
                    valid_loss = []

                    for tree in tree_list:
                        try:
                            generated_outputs = tree.val(input_points)
                        except:
                            continue

                        if label_outputs is None:
                            label_outputs = trees[i].val(input_points)
                            assert np.isfinite(label_outputs).all()
                        try:
                            if np.isfinite(generated_outputs).all():
                                valid_loss.append(
                                    np.sqrt(
                                        np.sum((generated_outputs - label_outputs) ** 2)
                                        / (np.sum(label_outputs**2) + eps)
                                    )
                                )
                        except:
                            continue

                    if len(valid_loss) > 0:
                        # generated tree is valid, compute other metrics
                        min_loss = min(valid_loss)
                        text_valid += 1
                        text_loss += min_loss

                        if params.print_outputs:
                            logger.info(
                                "[{}] Type: {} | Rel loss: {:.4f} | Text loss: {:.4f}".format(
                                    i, cur_type, rel_loss, min_loss
                                )
                            )
                            logger.info("Target:    {}".format(trees[i]))
                            try:
                                logger.info("Generated: {}\n".format(tree_list[0]))
                            except:
                                # logger.info("Generated: {}\n".format(tree_list[1]))
                                pass

                        if params.text_ode_solve:
                            # use tree + ODE solver
                            IC = samples["data"][i][0, :dim]

                            text_data_output = ODE_solver(
                                tree_list[0], IC, output_grid, logger=logger, type=samples["type"][i]
                            )

                            if text_data_output != None:
                                data_loss_valid += rel_loss
                                (
                                    _,
                                    text_rel_loss,
                                    _,
                                    _,
                                ) = compute_losses(
                                    text_data_output,
                                    data_seqs[i][input_len:, :],
                                    params.t_num - input_len,
                                    eps,
                                )
                                text_data_loss += text_rel_loss
                                text_valid_output += 1

            if not self.params.use_wandb:
                pbar.update(bs)

        data_loss = np.sum(np.array(data_loss))
        abs_data_loss = np.sum(np.array(abs_data_loss))
        data_loss_first_half = np.sum(np.array(data_loss_first_half))
        data_loss_second_half = np.sum(np.array(data_loss_second_half))

        if params.multi_gpu:
            # sync results on all gpus

            lst_sync = torch.Tensor(
                [
                    text_valid,
                    text_total,
                    text_loss,
                    data_loss,
                    abs_data_loss,
                    data_loss_first_half,
                    data_loss_second_half,
                ]
            ).cuda()
            total_loss = total_loss.cuda()
            total_abs_loss = total_abs_loss.cuda()
            total_count = total_count.cuda()

            dist.barrier()
            dist.all_reduce(lst_sync, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_abs_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)

            text_valid = lst_sync[0].item()
            text_total = lst_sync[1].item()
            text_loss = lst_sync[2].item()
            data_loss = lst_sync[3].item()
            abs_data_loss = lst_sync[4].item()
            data_loss_first_half = lst_sync[5].item()
            data_loss_second_half = lst_sync[6].item()

        if not params.text_only:
            s = "Rel loss - "
            for i, cur_type in enumerate(self.types):
                cur_loss = total_loss[i].item()
                cur_count = total_count[i].item()
                s += "{}: {:.6f}/{} \t ".format(cur_type, cur_loss / max(cur_count, 1), cur_count)

            logger.info(s)

            s = "Abs loss - "
            for i, cur_type in enumerate(self.types):
                cur_loss = total_abs_loss[i].item()
                cur_count = total_count[i].item()
                s += "{}: {:.6f}/{} \t ".format(cur_type, cur_loss / max(cur_count, 1), cur_count)

            logger.info(s)

        if params.text_ode_solve:
            logger.info(
                "Valid text - Data loss: {:.6f} - Data loss from text: {:.6f} - Text valid: {} - Text valid output: {}".format(
                    data_loss_valid / max(text_valid_output, 1),
                    text_data_loss / max(text_valid_output, 1),
                    text_valid,
                    text_valid_output,
                )
            )

        # logger.info("text_total: {} | Eval size per gpu: {}".format(text_total, eval_size_per_gpu))
        eval_size_per_gpu = text_total

        data_loss /= eval_size_per_gpu
        valid_fraction = text_valid / text_total
        text_loss /= max(text_valid, 1)
        abs_data_loss /= eval_size_per_gpu
        data_loss_first_half /= eval_size_per_gpu
        data_loss_second_half /= eval_size_per_gpu

        output = {
            "valid_fraction": valid_fraction,
            "text_loss": text_loss,
            "data_loss": data_loss,
            "data_loss_abs": abs_data_loss,
            "data_loss_first_half": data_loss_first_half,
            "data_loss_second_half": data_loss_second_half,
        }

        return output
