import json
import os
import io
import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from .optim import get_optimizer
from .utils import to_cuda
from collections import defaultdict
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt


logger = getLogger()


class LoadParameters(object):
    def __init__(self, modules, params):
        self.modules = modules
        self.params = params
        self.set_parameters()

    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        named_params = []
        for v in self.modules.values():
            named_params.extend([(k, p) for k, p in v.named_parameters() if p.requires_grad])
        self.parameters["model"] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def reload_checkpoint(self, path=None, root=None, requires_grad=True):
        """
        Reload a checkpoint if we find one.
        """
        if path is None:
            path = "checkpoint.pth"
        if root is None:
            root = self.params.dump_path
        checkpoint_path = os.path.join(root, path)

        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == "":
                return
            else:
                checkpoint_path = self.params.reload_checkpoint + "/checkpoint.pth"
                assert os.path.isfile(checkpoint_path)

        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location="cpu")

        # reload model parameters
        for k, v in self.modules.items():
            try:
                weights = data[k]
                v.load_state_dict(weights)
            except RuntimeError:  # remove the 'module.'
                weights = {name.partition(".")[2]: v for name, v in data[k].items()}
                v.load_state_dict(weights)
            v.requires_grad = requires_grad


class Trainer(object):
    def __init__(self, modules, env, params, path=None, root=None):
        """
        Initialize trainer.
        """

        # modules / params
        self.modules = modules
        self.params = params
        self.env = env

        # epoch / iteration size
        self.n_steps_per_epoch = params.n_steps_per_epoch
        self.inner_epoch = self.total_samples = self.n_equations = 0
        self.infos_statistics = defaultdict(list)
        self.errors_statistics = defaultdict(int)

        # data iterators
        self.iterators = {}

        # set parameters
        self.set_parameters()

        assert params.amp >= 0 or params.accumulate_gradients == 1
        if params.multi_gpu:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            for k in self.modules.keys():
                self.modules[k] = nn.parallel.DistributedDataParallel(
                    self.modules[k],
                    device_ids=[params.local_rank],
                    output_device=params.local_rank,
                    broadcast_buffers=True,
                    # find_unused_parameters=True,
                )

        # set optimizer
        self.set_optimizer()

        # float16 / distributed (AMP)
        self.scaler = None
        if params.amp >= 0:
            assert not self.params.cpu
            self.scaler = torch.cuda.amp.GradScaler()

        # stopping criterion used for early stopping
        if params.stopping_criterion != "":
            split = params.stopping_criterion.split(",")
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == "_":
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(",") if m != ""]
        for m in metrics:
            m = (m, False) if m[0] == "_" else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-np.infty if biggest else np.infty) for (metric, biggest) in self.metrics}

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.stats = OrderedDict(sum([[(x, []), (f"{x}-AVG-STOP-PROBS", [])] for x in env.TRAINING_TASKS], []))
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint(path=path, root=root)

        # file handler to export data
        if params.export_data:
            assert params.reload_data == ""
            params.export_path_prefix = os.path.join(params.dump_path, "data.prefix")
            self.file_handler_prefix = io.open(params.export_path_prefix, mode="a", encoding="utf-8")
            logger.info(f"Data will be stored in prefix in: {params.export_path_prefix} ...")

        # reload exported data
        if params.reload_data != "":
            logger.info(params.reload_data)
            assert params.export_data is False
            s = [x.split(",") for x in params.reload_data.split(";") if len(x) > 0]
            assert len(s) >= 1

            self.data_path = {
                task: (
                    train_path if train_path != "" else None,
                    valid_path if valid_path != "" else None,
                    test_path if test_path != "" else None,
                )
                for task, train_path, valid_path, test_path in s
            }

            logger.info(self.data_path)

            for task in self.env.TRAINING_TASKS:
                assert (task in self.data_path) == (task in params.tasks)
        else:
            self.data_path = None

        # create data loaders
        if not params.eval_only:
            if params.env_base_seed < 0:
                params.env_base_seed = np.random.randint(1_000_000_000)
            self.my_dataloaders = {
                task: self.env.create_train_iterator(task, self.data_path, params) for task in params.tasks
            }
            self.dataloader = {k: iter(v) for (k, v) in self.my_dataloaders.items()}

        self.t_eval = torch.from_numpy(env.generator.t_eval.astype(np.single))

        self.data_loss_weight = self.params.data_loss_weight
        self.data_loss = 0.0
        self.text_loss = 0.0

    def set_new_train_iterator_params(self, args={}):
        params = self.params
        if params.env_base_seed < 0:
            params.env_base_seed = np.random.randint(1_000_000_000)
        self.dataloader = {
            task: iter(self.env.create_train_iterator(task, self.data_path, params, args)) for task in params.tasks
        }
        logger.info("Succesfully replaced training iterator with following args:{}".format(args))
        return

    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        named_params = []
        for v in self.modules.values():
            named_params.extend([(k, p) for k, p in v.named_parameters() if p.requires_grad])
        self.parameters["model"] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            num = sum([torch.numel(p) for p in v])
            logger.info("Found %i parameters in %s." % (num, k))
            assert len(v) >= 1

    def set_optimizer(self):
        """
        Set optimizer.
        """
        params = self.params
        self.optimizer = get_optimizer(self.parameters["model"], params.lr, params.optimizer)
        logger.info("Optimizer: %s" % type(self.optimizer))

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            exit()

        params = self.params

        # optimizer
        optimizer = self.optimizer

        # regular optimization
        if params.amp == -1:
            optimizer.zero_grad()
            loss.backward()
            if params.clip_grad_norm > 0:
                clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)
            optimizer.step()

        else:
            if params.accumulate_gradients > 1:
                loss = loss / params.accumulate_gradients
            self.scaler.scale(loss).backward()

            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                if params.clip_grad_norm > 0:
                    self.scaler.unscale_(optimizer)
                    clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % self.params.print_freq != 0:
            return

        # s_total_eq = "- Total Eq: " + "{:.2e}".format(self.n_equations)
        s_iter = "%7i - " % self.n_total_iter
        s_stat = " || ".join(
            [
                "{}: {:7.4f}".format(k.upper().replace("_", "-"), np.mean(v))
                for k, v in self.stats.items()
                if type(v) is list and len(v) > 0
            ]
        )
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rates
        s_lr = (" - LR: ") + " / ".join("{:.4e}".format(group["lr"]) for group in self.optimizer.param_groups)

        max_mem = torch.cuda.max_memory_allocated() / 1024**2
        s_mem = " MEM: {:.2f} MB - ".format(max_mem)

        # log stats + learning rate
        logger.info(s_iter + s_mem + s_stat + s_lr)

    def get_generation_statistics(self, task):
        total_eqs = sum(x.shape[0] for x in self.infos_statistics[list(self.infos_statistics.keys())[0]])
        logger.info("Generation statistics (to generate {} eqs):".format(total_eqs))

        all_infos = defaultdict(list)
        for info_type, infos in self.infos_statistics.items():
            all_infos[info_type] = torch.cat(infos).tolist()
            infos = [torch.bincount(info) for info in infos]
            max_val = max([info.shape[0] for info in infos])
            aggregated_infos = torch.cat(
                [F.pad(info, (0, max_val - info.shape[0])).unsqueeze(-1) for info in infos],
                -1,
            ).sum(-1)
            non_zeros = aggregated_infos.nonzero(as_tuple=True)[0]
            vals = [
                (
                    non_zero.item(),
                    "{:.2e}".format((aggregated_infos[non_zero] / aggregated_infos.sum()).item()),
                )
                for non_zero in non_zeros
            ]
            logger.info("{}: {}".format(info_type, vals))
        all_infos = pd.DataFrame(all_infos)
        g = sns.PairGrid(all_infos)
        g.map_upper(sns.scatterplot)
        g.map_lower(sns.kdeplot, fill=True)
        g.map_diag(sns.histplot, kde=True)
        plt.savefig(os.path.join(self.params.dump_path, "statistics_{}.png".format(self.epoch)))

        str_errors = "Errors ({} eqs)\n ".format(total_eqs)
        for error_type, count in self.errors_statistics.items():
            str_errors += "{}: {}, ".format(error_type, count)
        logger.info(str_errors[:-2])
        self.errors_statistics = defaultdict(int)
        self.infos_statistics = defaultdict(list)

    def save_checkpoint(self, name, include_optimizer=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return

        if self.params.export_data:
            return

        path = os.path.join(self.params.dump_path, "%s.pth" % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            "epoch": self.epoch,
            "n_total_iter": self.n_total_iter,
            "best_metrics": self.best_metrics,
            "best_stopping_criterion": self.best_stopping_criterion,
            "params": {k: v for k, v in self.params.__dict__.items()},
        }

        for k, v in self.modules.items():
            # logger.warning(f"Saving {k} parameters ...")
            data[k] = v.state_dict()

        if include_optimizer:
            data["optimizer"] = self.optimizer.state_dict()
            if self.scaler is not None:
                data["scaler"] = self.scaler.state_dict()
            logger.warning(f"Saving model and optimizer parameters ...")
        else:
            logger.warning(f"Saving model parameters ...")

        torch.save(data, path)

    def reload_checkpoint(self, path=None, root=None, requires_grad=True):
        """
        Reload a checkpoint if we find one.
        """
        if path is None:
            path = "checkpoint.pth"

        if self.params.reload_checkpoint != "":
            checkpoint_path = os.path.join(self.params.reload_checkpoint, path)
            assert os.path.isfile(checkpoint_path)
        else:
            if root is not None:
                checkpoint_path = os.path.join(root, path)
            else:
                checkpoint_path = os.path.join(self.params.dump_path, path)
            if not os.path.isfile(checkpoint_path):
                logger.warning("Checkpoint path does not exist, {}".format(checkpoint_path))
                return

        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location="cpu")

        # reload model parameters
        for k, v in self.modules.items():
            weights = data[k]
            try:
                weights = data[k]
                v.load_state_dict(weights)
            except RuntimeError:  # remove the 'module.'
                weights = {name.partition(".")[2]: v for name, v in data[k].items()}
                v.load_state_dict(weights)
            v.requires_grad = requires_grad

        # reload optimizer
        if self.params.amp == -1:
            logger.warning("Reloading checkpoint optimizer ...")
            self.optimizer.load_state_dict(data["optimizer"])

        if "scaler" in data and self.scaler is not None:
            logger.warning("Reloading gradient scaler ...")
            self.scaler.load_state_dict(data["scaler"])

        # reload main metrics
        self.epoch = data["epoch"] + 1
        self.n_total_iter = data["n_total_iter"]
        self.best_metrics = data["best_metrics"]
        self.best_stopping_criterion = data["best_stopping_criterion"]
        logger.warning(f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ...")

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if self.params.save_periodic > 0 and self.epoch > 0 and self.epoch % self.params.save_periodic == 0:
            self.save_checkpoint("periodic-%i" % self.epoch)

    def save_best_model(self, scores, prefix=None, suffix=None):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            _metric = metric
            if prefix is not None:
                _metric = prefix + "_" + _metric
            if suffix is not None:
                _metric = _metric + "_" + suffix
            if _metric not in scores:
                logger.warning('Metric "%s" not found in scores!' % _metric)
                continue
            factor = 1 if biggest else -1

            if metric in self.best_metrics:
                best_so_far = factor * self.best_metrics[metric]
            else:
                best_so_far = -np.inf
            if factor * scores[_metric] > best_so_far:
                self.best_metrics[metric] = scores[_metric]
                logger.info("New best score for %s: %.6f" % (metric, scores[_metric]))
                self.save_checkpoint("best-%s" % metric)

    def end_epoch(self, scores=None):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None and (
            self.params.is_master or not self.stopping_criterion[0].endswith("_mt_bleu")
        ):
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info("New best validation score: %f" % self.best_stopping_criterion)
                self.decrease_counts = 0
            else:
                logger.info(
                    "Not a better validation score (%i / %i)." % (self.decrease_counts, self.decrease_counts_max)
                )
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info(
                    "Stopping criterion has been below its best value for more "
                    "than %i epochs. Ending the experiment..." % self.decrease_counts_max
                )
                if self.params.multi_gpu and "SLURM_JOB_ID" in os.environ:
                    os.system("scancel " + os.environ["SLURM_JOB_ID"])
                exit()

        self.save_checkpoint("checkpoint")
        self.epoch += 1

    def get_batch(self, task):
        """
        Return a training batch for a specific task.
        """
        try:
            batch, errors = next(self.dataloader[task])
        except:
            self.dataloader[task] = iter(self.my_dataloaders[task])
            batch, errors = next(self.dataloader[task])
        return batch, errors

    def export_data(self, task):
        """
        Export data to the disk.
        """
        samples, _ = self.get_batch(task)

        processed_e = len(samples["data"])
        for i in range(processed_e):
            data = samples["data"][i].tolist()  # save data as list of floats

            outputs = dict()
            outputs["type"] = samples["type"][i]
            outputs["data"] = data
            outputs["tree_encoded"] = samples["tree_encoded"][i]

            self.file_handler_prefix.write(json.dumps(outputs) + "\n")
            self.file_handler_prefix.flush()

        # number of processed sequences / words
        self.inner_epoch += 1
        self.n_equations += self.params.batch_size
        self.total_samples += self.params.batch_size
        # self.stats["processed_e"] += len(samples)

    def enc_dec_step(self, task):
        """
        Encoding / decoding step.
        """
        params = self.params

        if "embedder" in self.modules:
            embedder = self.modules["embedder"]
            embedder.train()

        if "text_encoder" in self.modules:
            text_encoder = self.modules["text_encoder"]
            text_encoder.train()

        if "text_decoder" in self.modules:
            text_decoder = self.modules["text_decoder"]
            text_decoder.train()

        if "data_encoder" in self.modules:
            data_encoder = self.modules["data_encoder"]
            data_encoder.train()

        if "data_decoder" in self.modules:
            data_decoder = self.modules["data_decoder"]
            data_decoder.train()

        if "fusion" in self.modules:
            fusion = self.modules["fusion"]
            fusion.train()

        env = self.env

        samples, _ = self.get_batch(task)

        # prepare data part

        t_eval = self.t_eval  # entire time sequence
        input_len = params.input_len
        assert t_eval.dtype == torch.float32

        data = samples["data"]

        if self.params.train_noise_gamma > 0:
            # add noise to data
            gamma = self.params.train_noise_gamma
            if self.params.noise_type == "multiplicative":
                for i, seq in enumerate(data):
                    data[i] = seq + (gamma * torch.abs(seq) * torch.randn_like(seq))
            else:  # additive
                eps = 1e-6
                for i, seq in enumerate(data):
                    cur_noise = torch.randn_like(seq)
                    sigma = gamma * torch.linalg.vector_norm(seq) / (torch.linalg.vector_norm(cur_noise) + eps)
                    data[i] = seq + sigma * cur_noise

        data_input, data_label, data_len, data_dim = env.batch_data_operator(data, t_eval, params.input_step)
        # data_input size: (slen/2, bs, output_dim+1)
        # data_label size: (slen, bs, output_dim)

        data_adim = torch.arange(params.max_output_dimension, dtype=torch.long, device=data_dim.device)
        data_pred_mask = data_adim[None] < (data_dim[:, None])  # (bs, output_dim)
        data_y = data_label[input_len:, :, :]
        data_pred_mask = data_pred_mask[None].expand_as(data_y)

        eps = 1e-5

        # reweight by Linfinity norm
        # loss_weight, _ = torch.max(torch.abs(data_y), dim=0, keepdim=True)  # (1, bs, output_dim)

        # reweight by L2 norm squared
        loss_weight = torch.linalg.vector_norm(data_y, dim=(0, -1), keepdim=True) ** 2  # (1, bs, 1)
        loss_weight = (
            (torch.reciprocal(loss_weight + eps) / np.single(data_label.size(1)))
            .expand_as(data_y)[data_pred_mask]
            .float()
        )

        data_y = data_y[data_pred_mask]

        assert len(data_y) == data_dim.sum().item() * (params.t_num - input_len)

        # cuda
        data_input, data_len, data_y, t_eval = to_cuda(data_input, data_len, data_y, t_eval)
        loss_weight = to_cuda(loss_weight)[0]

        if not params.no_text:
            # prepare text part

            text_input, text_len = env.batch_equations(
                self.env.word_to_idx(samples["tree_encoded"], float_input=False)
            )  # input for text decoder has <BOS> and <EOS>

            # target words to predict
            text_alen = torch.arange(text_len.max(), dtype=torch.long, device=text_len.device)
            text_pred_mask = text_alen[:, None] < (
                text_len[None] - 1
            )  # do not predict anything given the last target word (ignore <EOS>)

            text_y = text_input[1:].masked_select(text_pred_mask[:-1])  # target for text
            assert len(text_y) == (text_len - 1).sum().item()
            # cuda
            text_input, text_len, text_y = to_cuda(text_input, text_len, text_y)

            # input for text encoder has no <BOS> or <EOS>
            if params.use_skeleton:
                text_input_encoder, text_len_encoder = env.batch_equations_placeholder(
                    self.env.word_to_idx(samples["tree_skeleton"], float_input=False)
                )
                text_input_encoder, text_len_encoder = to_cuda(text_input_encoder, text_len_encoder)
            else:
                # if use complete text input, just delete <BOS>/<EOS>
                text_input_encoder = text_input[1:-1, :]
                text_len_encoder = text_len - 2

        # forward / loss

        with torch.cuda.amp.autocast(enabled=(params.amp >= 0), dtype=torch.bfloat16):
            # data features (data_len, bs, dim)
            data_input = embedder(data_input)
            data_input_encoded = data_encoder("fwd", x=data_input, lengths=data_len, causal=False)

            if params.no_text:
                fused_features_data = data_input_encoded
                text_len_encoder = None
            else:
                text_encoded = text_encoder("fwd", x=text_input_encoder, lengths=text_len_encoder, causal=False)

                fused_features_data = fusion(
                    "fwd",
                    x_data=data_input_encoded,
                    x_text=text_encoded,
                    lengths_data=data_len,
                    lengths_text=text_len_encoder,
                    causal=False,
                )  # (slen, bs, dim)

            """ 
            Task 1: data prediction
            
            INPUTS: first half data + text guess
            OUTPUTS: next half data
            
            """
            if not self.params.text_only:
                query_emb = data_decoder("query_emb", query_times=t_eval[input_len:])  # query locations (data_len, dim)

                data_decoded = data_decoder(
                    "fwd",
                    query_emb=query_emb,
                    src_enc=fused_features_data.transpose(0, 1),
                    src_len=(data_len, text_len_encoder),
                )
                _, data_loss = data_decoder(
                    "predict",
                    tensor=data_decoded,
                    pred_mask=data_pred_mask,
                    y=data_y,
                    weight=loss_weight,
                )

            """
            Task 2: text refinement

            INPUTS: first half data + text guess
            OUTPUTS: better text

            """
            if not self.params.data_only and not self.params.no_text:
                fused_features_text = fused_features_data

                text_decoded = text_decoder(
                    "fwd",
                    x=text_input,
                    lengths=text_len,
                    causal=True,
                    src_enc=fused_features_text.transpose(0, 1),
                    src_len=(data_len, text_len_encoder),
                )  # (slen, bs, dim)

                _, text_loss = text_decoder(
                    "predict",
                    tensor=text_decoded,
                    pred_mask=text_pred_mask,
                    y=text_y,
                    get_scores=False,
                )

        # add loss together, can possible add weights

        if self.params.text_only:
            loss = text_loss
            text_loss_item = text_loss.item()
            data_loss_item = 0.0
        elif self.params.data_only or self.params.no_text:
            loss = data_loss
            data_loss_item = data_loss.item()
            text_loss_item = 0.0
        else:
            data_loss_item = data_loss.item()
            text_loss_item = text_loss.item()

            loss = self.data_loss_weight * data_loss + text_loss

        self.stats[task].append(loss.item())
        self.data_loss += data_loss_item
        self.text_loss += text_loss_item

        # optimize
        self.optimize(loss)

        self.inner_epoch += 1
