from logging import getLogger
import os
import torch
import torch.distributed as dist
import numpy as np
from copy import deepcopy
from plot import plot,plot_sample_output,plot_one,plot_sample_output_noerror

import symbolicregression
from symbolicregression.model.model_wrapper import ModelWrapper
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from tabulate import tabulate

from tqdm import tqdm
import h5py
from sklearn.metrics import r2_score

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


def compute_losses(output, target, output_len, eps, dx):
    """
    output: (output_len, dim) or (output_len, x_grid_size, dim)
    target: (output_len, dim) or (output_len, x_grid_size, dim)

    RETURN: MSE, relative SE, first half relative SE, second half relative SE, mean_relative SE
    """
    half_len = output_len // 2
    target_first_half_sum = torch.sum(target[:half_len] ** 2)
    target_second_half_sum = torch.sum(target[half_len:] ** 2)
    first_half_diff = torch.sum((output[:half_len] - target[:half_len]) ** 2)
    second_half_diff = torch.sum((output[half_len:] - target[half_len:]) ** 2)
    # total_sum = torch.sum(target ** 2)

    # abs_loss = torch.sum((output - target) ** 2)
    # rel_loss = torch.sqrt(abs_loss / (eps + total_sum))
    abs_loss = first_half_diff + second_half_diff
    rel_loss = torch.sqrt(abs_loss / ( target_first_half_sum + target_second_half_sum))
    mean_rel_loss = torch.sqrt(abs_loss / ( torch.sum((target - target.mean()) ** 2)))

    l1_loss = torch.sum(torch.abs(output - target)) / ( torch.sum(torch.abs(target)))

    rel_loss_first_half = torch.sqrt(first_half_diff / (target_first_half_sum))
    rel_loss_second_half = torch.sqrt(second_half_diff / (target_second_half_sum))

    x_grid_size = 1 if output.dim() == 2 else output.size(-2)

    return (
        abs_loss / output_len / output.size(-1) / x_grid_size,
        rel_loss,
        rel_loss_first_half,
        rel_loss_second_half,
        mean_rel_loss,
        l1_loss,
    )


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
        self.space_dim = self.params.max_input_dimension

        self.t_eval = torch.from_numpy(self.env.generator.t_eval.astype(np.single))
        self.x_grid = torch.from_numpy(self.env.generator.x_grid.astype(np.single))
        if self.space_dim > 0:
            # self.x_grid = torch.from_numpy(self.env.generator.x_grid.astype(np.single))
            self.x_grid_size = self.env.generator.x_grid_size
            # self.input_tx_grid = self.env.get_tx_grid(
            #     self.t_eval[0 : self.params.input_len], self.x_grid, self.space_dim, self.x_grid_size
            # )  # (t_num*x_num, 1+space_dim)
            # self.output_tx_grid = self.env.get_tx_grid(
            #     self.t_eval[self.params.input_len :], self.x_grid, self.space_dim, self.x_grid_size
            # ).cuda()
        else:
            # self.x_grid = None
            self.x_grid_size = 1
            # self.input_tx_grid = None
            # self.output_tx_grid = None

        # if self.params.ode_gen:
        #     self.types = self.env.generator.ode_generator.types
        # else:
        #     self.types = self.env.generator.pde_generator.types
        # if str(self.params.eval_types) != "":
        #     if str(self.params.eval_types).startswith("pde"):
        #         self.types = self.env.generator.pde_generator.eval_types
        #     else:
        #         self.types = self.env.generator.ode_generator.eval_types
        # else:
        self.all_types = [
            "heat",
            "porous_medium",
            "advection",
            "kdv",
            "fplanck",
            "diff_logisreact_1D",
            "diff_linearreact_1D",
            "diff_bistablereact_1D",
            "diff_squarelogisticreact_1D",
            "burgers",
            "conservation_linearflux",
            "conservation_sinflux",
            "conservation_cosflux",
            "conservation_cubicflux",
            "inviscid_burgers",
            "inviscid_conservation_sinflux",
            "inviscid_conservation_cosflux",
            "inviscid_conservation_cubicflux",
            "cahnhilliard_1D",
            "wave",
            "Klein_Gordon",
            "Sine_Gordon",
        ]
        self.alltypes_to_idx = {s: i for i, s in enumerate(self.all_types)}
        if str(self.params.types).startswith("pde"):
            self.types = self.env.generator.pde_generator.types
        else:
            self.types = self.env.generator.ode_generator.types
        self.types_to_idx = {s: i for i, s in enumerate(self.types)}

        if self.space_dim == 0:
            self.input_points = np.random.uniform(-5, 5, size=(self.eval_num_input_points, self.output_dim, 1))
        else:
            t_grid = np.linspace(0.0, self.params.t_range, self.params.t_num)
            x_grid = np.linspace(0.0, self.params.x_range, self.params.x_num)
            coeff = np.random.uniform(-5, 5, size=(8, self.params.max_input_dimension))
            # Create mesh grids
            T, X = np.meshgrid(t_grid, x_grid, indexing="ij")

            # Calculate the terms using vectorized operations
            input_points = np.zeros((self.params.max_input_dimension, self.params.t_num, self.params.x_num, 8))
            for i in range(self.params.max_input_dimension):
                input_points[i, :, :, 0] = (coeff[0, i] + coeff[1, i] * T + coeff[2, i] * T**2) * (
                    coeff[3, i] + coeff[4, i] * X + coeff[5, i] * X**2 + coeff[6, i] * X**3 + coeff[7, i] * X**4
                )
                input_points[i, :, :, 1] = (coeff[1, i] + 2 * coeff[2, i] * T) * (
                    coeff[3, i] + coeff[4, i] * X + coeff[5, i] * X**2 + coeff[6, i] * X**3 + coeff[7, i] * X**4
                )
                # ut
                input_points[i, :, :, 2] = (2 * coeff[2, i]) * (
                    coeff[3, i] + coeff[4, i] * X + coeff[5, i] * X**2 + coeff[6, i] * X**3 + coeff[7, i] * X**4
                )
                # utt
                input_points[i, :, :, 3] = (coeff[0, i] + coeff[1, i] * T + coeff[2, i] * T**2) * (
                    coeff[4, i] + 2 * coeff[5, i] * X + 3 * coeff[6, i] * X**2 + 4 * coeff[7, i] * X**3
                )  # ux
                input_points[i, :, :, 4] = (coeff[0, i] + coeff[1, i] * T + coeff[2, i] * T**2) * (
                    2 * coeff[5, i] + 6 * coeff[6, i] * X + 12 * coeff[7, i] * X**2
                )  # uxx
                input_points[i, :, :, 5] = (coeff[0, i] + coeff[1, i] * T + coeff[2, i] * T**2) * (
                    6 * coeff[6, i] + 24 * coeff[7, i] * X
                )  # uxxx
                input_points[i, :, :, 6] = (coeff[0, i] + coeff[1, i] * T + coeff[2, i] * T**2) * (
                    24 * coeff[7, i]
                )  # uxxxx
                input_points[i, :, :, 7] = X  # x
            # input_points = np.zeros((6, params.t_num,params.x_num ))
            # for i in range( params.t_num):
            #     for j in range(params.x_num):
            #         input_points[0,i,j] = (coeff[0]+coeff[1]*t_grid[i]+coeff[2]*x_grid[j]+coeff[3]*t_grid[i]*x_grid[j]+coeff[4]*t_grid[i]**2+coeff[5]*x_grid[j]**2)
            #         input_points[1,i,j] = (coeff[1]+coeff[3]*x_grid[j]+coeff[4]*t_grid[i]*2)
            #         input_points[2,i,j] = (coeff[2]+coeff[3]*t_grid[i]+coeff[5]*x_grid[j]*2)
            #         input_points[3,i,j] = (coeff[3])
            #         input_points[4,i,j] = (coeff[4])
            #         input_points[5,i,j] = (coeff[5])
            # input_points = np.random.uniform(-5, 5, size=(self.eval_num_input_points, self.space_dim))
            self.input_points = input_points

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

        if "normalizer" in self.modules:
            normalizer = self.modules["normalizer"].module if params.multi_gpu else self.modules["normalizer"]
            normalizer.eval()
        else:
            normalizer = None

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
            normalizer=normalizer,
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
            output_step=params.eval_output_step,
            amp=params.eval_amp,
            space_dim=self.space_dim,
            # input_tx_grid=self.input_tx_grid,
            # output_tx_grid=self.output_tx_grid,
            x_grid_size=self.x_grid_size,
        )

        total_loss = torch.zeros(len(self.types), dtype=torch.float32)
        best_total_loss = torch.zeros(len(self.types), dtype=torch.float32)
        min_data_loss = torch.ones(len(self.types), dtype=torch.float32)
        min_data_loss_index = torch.zeros(len(self.types), dtype=torch.float32)
        max_data_loss = torch.zeros(len(self.types), dtype=torch.float32)
        max_data_loss_index = torch.zeros(len(self.types), dtype=torch.float32)
        total_abs_loss = torch.zeros(len(self.types), dtype=torch.float32)
        total_count = torch.zeros(len(self.types), dtype=torch.long)
        total_mean_loss = torch.zeros(len(self.types), dtype=torch.float32)
        total_l1_loss = torch.zeros(len(self.types), dtype=torch.float32)
        best_l1_loss = torch.zeros(len(self.types), dtype=torch.float32)
        output_list = []
        target_list = []
        type_list = []
        if save:
            if save_file is None:
                save_file = (
                    self.params.eval_dump_path if self.params.eval_dump_path is not None else self.params.dump_path
                )
            if not os.path.exists(save_file):
                os.makedirs(save_file)

            # save_target = os.path.join(save_file, "target.h5")
            save_file = os.path.join(save_file, "eval_in_domain.h5")

        if not self.params.use_wandb:
            pbar = tqdm(total=eval_size_per_gpu)

        input_len = self.params.input_len
        t_eval = self.trainer.t_eval
        output_len = len(t_eval) - input_len
        input_points = self.input_points
        eval_output_start = self.params.eval_output_start
        eval_output_end = len(t_eval)
        text_loss = 0.0
        num_batches =0
        text_valid = 0
        text_total = 0
        eps = 1e-6

        data_loss = []
        #r2_losses = []
        l1_loss = []
        abs_data_loss = []
        data_loss_first_half = []
        data_loss_second_half = []
        data_loss_mean = []
        data_loss_type = [[] for _ in range(len(self.types))]
        abs_data_loss_type = [[] for _ in range(len(self.types))]
        l1_loss_type = [[] for _ in range(len(self.types))]

        if params.text_ode_solve:
            # use text output + ODE solver
            data_loss_valid = 0.0
            text_data_loss = 0.0
            text_valid_output = 0
            output_grid = env.generator.t_eval[eval_output_start : eval_output_end : params.eval_output_step]

        for samples, _ in iterator:
            data_seqs = samples["data"]  # (bs, data_len, output_dim) or (bs, data_len, x_grid_size, output_dim)

            if self.params.eval_noise_gamma > 0:
                # add noise on data
                if self.params.noise_type == "multiplicative":
                    for i, seq in enumerate(data_seqs):
                        data_seqs[i] = seq + (
                            gamma * torch.abs(seq) * torch.from_numpy(noise_rng.randn(*seq.size()).astype(np.single))
                        )
                else:  # additive
                    for i, seq in enumerate(data_seqs):
                        cur_noise = torch.from_numpy(noise_rng.randn(*seq.size()).astype(np.single))
                        sigma = gamma * torch.linalg.vector_norm(seq) / (torch.linalg.vector_norm(cur_noise) + eps)
                        data_seqs[i] = seq + sigma * cur_noise

            if params.use_skeleton:
                text_seqs = samples["tree_skeleton"]  # (bs, text_len)
            else:
                text_seqs = samples["tree_encoded"]  # (bs, text_len)

            dims = samples["dim"] if "dim" in samples else None

            trees = samples["tree"]  # (bs, )
            bs = len(data_seqs)
            text_total += bs
            num_batches +=1

            # if self.space_dim == 0:
            #     data_input = [seq[: input_len : params.input_step, :] for seq in data_seqs]
            # else:
            #     data_input = [seq[: input_len : params.input_step, :, :] for seq in data_seqs]
            data_input = [seq[: input_len : params.input_step] for seq in data_seqs]
            # norms = [torch.linalg.norm(seq) for seq in data_seqs]

            data_outputs, text_outputs = mw(
                data_input=data_input,  # (bs, input_len, output_dim) or (bs, input_len, x_grid_size, output_dim)
                text_input=text_seqs,  # (bs, text_len)
                dims=dims,
                logger=logger,
                eval_output_start=eval_output_start,
                eval_output_end=eval_output_end,
            )
            # data_outputs: torch tensor of shape (bs, data_len, output_dim) or (bs, data_len, x_grid_size, output_dim)
            # text_outputs: nested list of shape (bs, tree_nums), some inner lists are possibly empty, elements are trees

            # computing eval losses

            data_outputs = data_outputs.to(data_seqs[0].device)
            targets = [seq[eval_output_start : eval_output_end : params.eval_output_step] for seq in samples["data"]]
            if params.plot_comparison:
                plot(data_outputs, targets, samples["type"], params, plot_type=self.types)

                plot(
                    [torch.abs(data_outputs[i] - targets[i]) for i in range(len(data_seqs))],
                    None,
                    samples["type"],
                    params,
                    notes="_diff",
                    plot_type=self.types,
                )
                plot(data_seqs, None, samples["type"], params, notes="all", plot_type=self.types, initial=True)
            cur_output = []
            cur_target = []
            for i in range(bs):
                rel_loss = 0.0
                if not params.text_only:
                    cur_type = samples["type"][i]
                    # data loss
                    if "dim" in samples:
                        dim = samples["dim"][i]
                    else:
                        dim = data_seqs[i].size(-1)
                    if self.space_dim == 0:
                        (abs_loss, rel_loss, rel_loss_first_half, rel_loss_second_half, rel_loss_mean, rel_l1_loss) = (
                            compute_losses(
                                data_outputs[i, :, :dim],
                                data_seqs[i][eval_output_start : eval_output_end : params.eval_output_step, :dim],
                                (params.t_num - input_len) // params.eval_output_step,
                                eps,
                                params.x_range / params.x_num,
                            )
                        )
                        # this_output = np.insert(data_outputs[i, :, :dim].flatten(), 0, self.alltypes_to_idx[cur_type])
                        type_list.append(self.alltypes_to_idx[cur_type])
                        this_output = data_outputs[i, :, :dim]
                        output_list.append(this_output.flatten())
                        cur_output.append(this_output.flatten())
                        this_target = data_seqs[i][
                            eval_output_start : eval_output_end : params.eval_output_step, :dim
                        ]
                        target_list.append(this_target.flatten())
                        cur_target.append(this_target.flatten())
                    else:
                        (abs_loss, rel_loss, rel_loss_first_half, rel_loss_second_half, rel_loss_mean, rel_l1_loss) = (
                            compute_losses(
                                data_outputs[i, :, :, :dim],
                                data_seqs[i][eval_output_start : eval_output_end : params.eval_output_step, ..., :dim],
                                (params.t_num - input_len) // params.eval_output_step,
                                eps,
                                params.x_range / params.x_num,
                            )
                        )
                        type_list.append(self.alltypes_to_idx[cur_type])
                        this_output = data_outputs[i, :, :,:dim]
                        output_list.append(this_output.flatten())
                        cur_output.append(this_output.flatten())
                        this_target = data_seqs[i][
                            eval_output_start : eval_output_end : params.eval_output_step, ..., :dim
                        ]
                        target_list.append(this_target.flatten())
                        cur_target.append(this_target.flatten())

                    abs_data_loss.append(abs_loss)
                    data_loss.append(rel_loss)
                    data_loss_first_half.append(rel_loss_first_half)
                    data_loss_second_half.append(rel_loss_second_half)
                    data_loss_mean.append(rel_loss_mean)
                    l1_loss.append(rel_l1_loss)

                    abs_data_loss_type[self.types_to_idx[cur_type]].append(abs_loss)
                    data_loss_type[self.types_to_idx[cur_type]].append(rel_loss)
                    total_l1_loss[self.types_to_idx[cur_type]] += rel_l1_loss
                    l1_loss_type[self.types_to_idx[cur_type]].append(rel_l1_loss)

                    total_loss[self.types_to_idx[cur_type]] += rel_loss
                    total_abs_loss[self.types_to_idx[cur_type]] += abs_loss
                    total_mean_loss[self.types_to_idx[cur_type]] += rel_loss_mean
                    total_count[self.types_to_idx[cur_type]] += 1

                    if rel_loss < min_data_loss[self.types_to_idx[cur_type]]:
                        min_data_loss_index[self.types_to_idx[cur_type]] = i
                        min_data_loss[self.types_to_idx[cur_type]] = rel_loss
                    if rel_loss > max_data_loss[self.types_to_idx[cur_type]]:
                        max_data_loss_index[self.types_to_idx[cur_type]] = i
                        max_data_loss[self.types_to_idx[cur_type]] = rel_loss
                        if params.plot_worst:
                            for jj in range(self.space_dim):
                                title = "type_{}_dim{}_worst".format(cur_type, jj)
                                plot_sample_output_noerror(
                                    data_outputs[int(i)][:, :, jj],
                                    targets[int(i)][:, :, jj],
                                    self.x_grid[:, jj],
                                    self.t_eval[
                                    eval_output_start: eval_output_end: params.eval_output_step],
                                    params, title=title)
                                plot_sample_output(data_outputs[int(i)][:, :, jj],
                                                   targets[int(i)][:, :, jj],
                                                   self.x_grid[:, jj],
                                                   self.t_eval[
                                                   eval_output_start: eval_output_end: params.eval_output_step],
                                                   params, title=title)

                    # if params.print_outputs:
                    #    plot_dict =  plot([data_outputs[i]], [target[i]], cur_type, params, plot_type=self.types,plot_dict=plot_dict)
                    #     data = data_seqs[i]
                    #     output = data_outputs[i]
                    #     fig = plt.figure()
                    #     for j in range(dim):
                    #         c = colors[j]
                    #         plt.plot(t_eval, data[:, j], "--", linewidth=1.4, alpha=0.8, color=c, label=f"target {j}")
                    #         plt.plot(t_eval[input_len:], output[:, j], "-", linewidth=2, color=c, label=f"output {j}")
                    #     plt.legend(loc="best")
                    #     plt.title("{} | {} | {:.6f}".format(i, cur_type, rel_loss))
                    #     plt.savefig("figures/eval_{}.png".format(i))
                if not params.text_only and params.print_outputs and i%50 ==0:
                    for jj in range(self.space_dim):
                        title = "type_{}_dim{}_{}".format(cur_type, jj, i)
                        plot_sample_output(this_output[:, :, jj], this_target[:, :, jj],
                                           self.x_grid[:, jj],
                                           self.t_eval[
                                           eval_output_start: eval_output_end: params.eval_output_step],
                                           params, title=title)
                        title = "type_{}_dim{}_{}_shortinput".format(cur_type, jj, i)
                        plot_one(data_seqs[i][0: 4, ..., jj], params, title=title, input_size=1)
                        title = "type_{}_dim{}_{}_sampleoutput".format(cur_type, jj, i)
                        plot_one(data_seqs[i][4:: params.eval_output_step, ..., jj], params,
                                 title=title)
                        logger.info(
                            "[{}] Type: {} | Rel loss: {:.4f} ".format(
                                i, cur_type, rel_loss,
                            )
                        )
                        logger.info("Target:    {}".format(trees[i]))
                if not params.data_only and not params.no_text:
                    # text loss
                    tree_list = text_outputs[i]
                    label_outputs = None
                    valid_loss = []

                    for tree in tree_list:
                        try:
                            generated_outputs = tree.val(input_points, self.space_dim)

                        except:
                            continue

                        if label_outputs is None:
                            # if self.space_dim == 0:
                            #     label_outputs = tree[i].val(input_points)
                            # else:
                            #     label_outputs = tree[i].val(t_grid, x_grid,coeff)
                            label_outputs = trees[i].val(input_points, self.space_dim)
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

                    # logger.info("[{}] Input:     {}".format(i, text_seqs[i]))
                    # logger.info("[{}] Target:    {}".format(i, trees[i]))

                    if len(valid_loss) > 0:
                        # generated tree is valid, compute other metrics
                        min_loss = min(valid_loss)
                        text_valid += 1
                        text_loss += min_loss

                        if params.print_outputs:
                            if i % 50 == 0:
                                logger.info(
                                    "[{}] Type: {} | Rel loss: {:.4f} | Mean Rel loss:{:4f} |Text loss: {:.4f}".format(
                                        i, cur_type, rel_loss,rel_loss_mean, min_loss
                                    )
                                )
                                # logger.info("Input:     {}".format(text_seqs[i]))
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
                                    _,
                                    _,
                                ) = compute_losses(
                                    text_data_output,
                                    data_seqs[i][eval_output_start : eval_output_end : params.eval_output_step, :dim],
                                    (params.t_num - input_len) // params.eval_output_step,
                                    eps,
                                    params.x_range / params.x_num,
                                )
                                text_data_loss += text_rel_loss
                                text_valid_output += 1

            # r2_losses.append(r2_score(np.stack(cur_target), np.stack(cur_output)[:, 1:]))
           # r2_losses.append(r2_score(np.stack(cur_target), np.stack(cur_output)))

            if not self.params.use_wandb:
                pbar.update(bs)

        if save:
            with h5py.File(save_file, "w") as hf:
                save_output = np.stack(output_list)
                hf.create_dataset("output", data=save_output)
                save_target = np.stack(target_list)
                hf.create_dataset("target", data=save_target)
                save_type = np.array(type_list)
                hf.create_dataset("type", data=save_type)
                logger.info(
                    f"Output ({save_output.shape}), target ({save_target.shape}), types ({save_type.shape}) saved at: {save_file}"
                )

        data_loss = np.sum(np.array(data_loss))
        abs_data_loss = np.sum(np.array(abs_data_loss))
        data_loss_first_half = np.sum(np.array(data_loss_first_half))
        data_loss_second_half = np.sum(np.array(data_loss_second_half))
        data_loss_mean = np.sum(np.array(data_loss_mean))
       # print(r2_losses)
       # r2_losses = np.sum(np.array(r2_losses))
        l1_loss = np.sum(np.array(l1_loss))

        best_95perc_data_loss = 0
        best_95perc_l1_loss = 0
        for i in range(len(self.types)):
            cur_len = len(data_loss_type[i])
            best95_cur = np.sort(np.array(data_loss_type[i]))[: int(0.95 * cur_len)]
            best_total_loss[i] += np.sum(best95_cur)

            best_95perc_data_loss += np.sum(best95_cur)

            best95l1_cur = np.sort(np.array(l1_loss_type[i]))[: int(0.95 * cur_len)]
            best_l1_loss[i] += np.sum(best95l1_cur)

            best_95perc_l1_loss += np.sum(best95l1_cur)
        if params.multi_gpu:
            # sync results on all gpus

            lst_sync = torch.Tensor(
                [
                    text_valid,
                    text_total,
                    text_loss,
                    data_loss,
                    best_95perc_data_loss,
                    best_95perc_l1_loss,
                    abs_data_loss,
                    data_loss_first_half,
                    data_loss_second_half,
                    data_loss_mean,
            #        r2_losses,
                    l1_loss,
                ]
            ).cuda()
            total_loss = total_loss.cuda()
            total_abs_loss = total_abs_loss.cuda()
            total_count = total_count.cuda()
            total_mean_loss = total_mean_loss.cuda()
            total_l1_loss = total_l1_loss.cuda()

            dist.barrier()
            dist.all_reduce(lst_sync, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_abs_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_mean_loss, op=dist.ReduceOp.SUM)

            text_valid = lst_sync[0].item()
            text_total = lst_sync[1].item()
            text_loss = lst_sync[2].item()
            data_loss = lst_sync[3].item()
            best_95perc_data_loss = lst_sync[4].item()
            best_95perc_l1_loss = lst_sync[5].item()
            abs_data_loss = lst_sync[6].item()
            data_loss_first_half = lst_sync[7].item()
            data_loss_second_half = lst_sync[8].item()
            data_loss_mean = lst_sync[9].item()
           # r2_losses = lst_sync[9].item()
            l1_loss = lst_sync[10].item()

        if not params.text_only:
            # s = "Rel loss - "

            headers = []
            table = []
            # outputs = np.stack(output_list)[:, 1:]
            # labels = np.stack(output_list)[:, 0]
            outputs = np.stack(output_list)
            labels = np.array(type_list)
            target = np.stack(target_list)
            for i, cur_type in enumerate(self.types):
                cur_loss = total_loss[i].item()
                cur_count = total_count[i].item()
                cur_min_loss = min_data_loss[i].item()
                cur_min_loss_index = min_data_loss_index[i].item()
                cur_max_loss = max_data_loss[i].item()
                cur_max_loss_index = max_data_loss_index[i].item()
                cur_best_loss = best_total_loss[i].item()
                cur_best_l1_loss = best_l1_loss[i].item()
                cur_mean_loss = total_mean_loss[i].item()
                cur_l1_loss = total_l1_loss[i].item()

                cur_type_indx = self.alltypes_to_idx[cur_type]
                if np.sum(labels == cur_type_indx) != 0:
                    cur_r2_loss = r2_score(target[labels == cur_type_indx, :], outputs[labels == cur_type_indx, :])
                    mean = np.mean(target[labels == cur_type_indx, :], axis=0)
                    cur_mean_pred_loss = np.linalg.norm(target[labels == cur_type_indx, :] - mean) / np.linalg.norm(
                        target[labels == cur_type_indx, :]
                    )
                else:
                    cur_r2_loss = 0

                    cur_mean_pred_loss = 0

                headers.append("Type")
                table.append([cur_type])

                headers.append("Size")
                table[i].append(cur_count)

                headers.append("Rel L2")
                table[i].append(cur_loss / max(cur_count, 1))

                headers.append("Best 95% L2")
                table[i].append(cur_best_loss / max(0.95 * cur_count, 1))

                cur_abs_loss = total_abs_loss[i].item()

                headers.append("MSE")
                table[i].append(cur_abs_loss / max(cur_count, 1))

                headers.append("Mean Rel L2")
                table[i].append(cur_mean_loss / max(cur_count, 1))

                headers.append("L1 ")
                table[i].append(cur_l1_loss / max(cur_count, 1))

                headers.append("Best 95% L1")
                table[i].append(cur_best_l1_loss / max(0.95 * cur_count, 1))

                headers.append("R2 ")
                table[i].append(cur_r2_loss)

                headers.append("prediction by mean ")
                table[i].append(cur_mean_pred_loss)

                # s += "{}: {:.6f}/{}  best 95 perc: {:.6f} \n ".format(
                #     cur_type, cur_loss / max(cur_count, 1), cur_count, cur_best_loss / max(0.95 * cur_count, 1)
                # )

                if params.print_outputs:

                    # headers.append("Min")
                    # table[i].append(cur_min_loss)
                    #
                    # headers.append("Min idx")
                    # table[i].append(cur_min_loss_index)

                    headers.append("Max")
                    table[i].append(cur_max_loss)

                    headers.append("Max idx")
                    table[i].append(cur_max_loss_index)

                    # s += "min {} at {}, max {} at {}\t".format(
                    #     cur_min_loss, cur_min_loss_index, cur_max_loss, cur_max_loss_index
                    # )
                    for jj in range(self.space_dim):
                        title = "type_{}_dim{}_worst".format(cur_type, jj)
                        plot_sample_output_noerror(data_outputs[int(cur_max_loss_index)][:,:,jj], targets[int(cur_max_loss_index)][:,:,jj],
                                                   self.x_grid[:, jj],
                                                   self.t_eval[
                                                   eval_output_start: eval_output_end: params.eval_output_step],
                                                   params, title=title)
                        plot_sample_output(data_outputs[int(cur_max_loss_index)][:, :, jj],
                                                   targets[int(cur_max_loss_index)][:, :, jj],
                                                   self.x_grid[:, jj],
                                                   self.t_eval[
                                                   eval_output_start: eval_output_end: params.eval_output_step],
                                                   params, title=title)
                    if params.plot_comparison:
                        plot(
                            [data_outputs[int(cur_max_loss_index)]],
                            [targets[int(cur_max_loss_index)]],
                            [cur_type],
                            params,
                            notes="_worst_",
                            num_choice=1,
                            plot_type=[cur_type],
                        )
                        plot(
                            [torch.abs(data_outputs[int(cur_max_loss_index)] - targets[int(cur_max_loss_index)])],
                            None,
                            [cur_type],
                            params,
                            notes="_worstdiff_",
                            num_choice=1,
                            plot_type=[cur_type],
                        )

                    recomputed_loss = torch.norm(
                        data_outputs[int(cur_max_loss_index)] - targets[int(cur_max_loss_index)]
                    ) / torch.norm(targets[int(cur_max_loss_index)])

                    headers.append("Recomp abs loss")
                    table[i].append(torch.norm(data_outputs[int(cur_max_loss_index)] - targets[int(cur_max_loss_index)]))

                    headers.append("Recomp worst loss")
                    table[i].append(recomputed_loss)

                    headers.append("Recomp norm")
                    table[i].append(torch.norm(targets[int(cur_max_loss_index)]))

                    # s += "recomputed absloss {} , worstloss = {}, norm = {}\n".format(
                    #     torch.norm(data_outputs[int(cur_max_loss_index)] - target[int(cur_max_loss_index)]),
                    #     recomputed_loss,
                    #     torch.norm(target[int(cur_max_loss_index)]),
                    # )
            # logger.info(s)

            # s = "Abs loss - "
            # for i, cur_type in enumerate(self.types):
            #     cur_loss = total_abs_loss[i].item()
            #     cur_count = total_count[i].item()
            #     s += "{}: {:.6f}/{} \t ".format(cur_type, cur_loss / max(cur_count, 1), cur_count)

            # logger.info(s)

            logger.info("Evaluation Stats\n{}".format(tabulate(table, headers=headers, tablefmt="grid")))

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
        # best_95perc_data_loss/= 0.95 * eval_size_per_gpu
        valid_fraction = text_valid / text_total
        text_loss /= max(text_valid, 1)
        abs_data_loss /= eval_size_per_gpu
        data_loss_first_half /= eval_size_per_gpu
        data_loss_second_half /= eval_size_per_gpu
        data_loss_mean /= eval_size_per_gpu
        l1_loss /= eval_size_per_gpu
        r2_losses = r2_score(np.stack(target_list), np.stack(output_list))
        
       # print(r2_losses)
        if self.params.text_only:
            loss = text_loss
            if valid_fraction == 0.0:
                loss = np.inf
        elif self.params.data_only or self.params.no_text:
            loss = data_loss
        else:
            loss = self.params.data_loss_weight * data_loss + text_loss
            if valid_fraction == 0.0:
                loss = np.inf

        output = {
            "valid_fraction": valid_fraction,
            "text_loss": text_loss,
            "data_loss": data_loss,
            "data_loss_abs": abs_data_loss,
            "data_loss_first_half": data_loss_first_half,
            "data_loss_second_half": data_loss_second_half,
            "total_loss": loss,
            "data_loss_mean": data_loss_mean,
            "l1_loss": l1_loss,
            "r2_loss": r2_losses,
            # "best_data_loss": best_95perc_data_loss,
        }

        return output

    # def compute_metrics(self,
    #                     data_type,
    #                     task,
    # preds_path,labels_path):
    #     total_loss = torch.zeros(len(self.types), dtype=torch.float32)
    #     best_total_loss = torch.zeros(len(self.types), dtype=torch.float32)
    #     min_data_loss = torch.ones(len(self.types), dtype=torch.float32)
    #     min_data_loss_index = torch.zeros(len(self.types), dtype=torch.float32)
    #     max_data_loss = torch.zeros(len(self.types), dtype=torch.float32)
    #     max_data_loss_index = torch.zeros(len(self.types), dtype=torch.float32)
    #     total_abs_loss = torch.zeros(len(self.types), dtype=torch.float32)
    #     total_count = torch.zeros(len(self.types), dtype=torch.long)
    #     total_mean_loss = torch.zeros(len(self.types), dtype=torch.float32)
    #
    #     input_len = self.params.input_len
    #     t_eval = self.trainer.t_eval
    #     output_len = len(t_eval) - input_len
    #     input_points = self.input_points
    #     eval_output_start = self.params.eval_output_start
    #     eval_output_end = len(t_eval)
    #
    #     text_loss = 0.0
    #     text_valid = 0
    #     text_total = 0
    #     eps = 1e-6
    #
    #     env = getattr(self, "{}_env".format(data_type))
    #     eval_size_per_gpu = self.params.eval_size // self.params.n_gpu_per_node
    #     if self.dataloader is None:
    #         self.dataloader = env.create_test_iterator(
    #             data_type,
    #             task,
    #             data_path=self.trainer.data_path,
    #             batch_size=self.params.batch_size_eval,
    #             params=self.params,
    #             size=eval_size_per_gpu,
    #             test_env_seed=self.params.test_env_seed,
    #         )
    #
    #     iterator = self.dataloader
    #     for samples, _ in iterator:
    #         target =np.array([seq[eval_output_start : eval_output_end : self.params.eval_output_step] for seq in samples["data"]])
    #         prediction = np.array([])[:,1:]
    #         label_indx = np.array([])[:,0]
    #
