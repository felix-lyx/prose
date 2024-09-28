import os
from logging import getLogger
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from transformers import get_scheduler
from utils.misc import to_cuda
from dataset import get_dataset
from data_utils.collate import custom_collate
from dadaptation import DAdaptAdan

logger = getLogger()


class Trainer(object):
    def __init__(self, modules, params, symbol_env):
        """
        Initialize trainer.
        """

        # modules / params
        self.modules = modules
        self.params = params
        self.symbol_env = symbol_env

        # epoch / iteration size
        self.n_steps_per_epoch = params.n_steps_per_epoch
        self.inner_epoch = 0

        # set parameters
        self.set_parameters()

        # distributed
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

        # amp
        self.scaler = None
        if params.amp:
            self.scaler = torch.amp.GradScaler("cpu" if params.cpu else "cuda")

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

        # reload potential checkpoints
        self.reload_checkpoint()

        # create data loaders
        if not params.eval_only:
            self.dataloader_count = 0
            self.dataset = get_dataset(params, symbol_env, split="train")
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=params.batch_size,
                # shuffle=True,
                num_workers=params.num_workers,
                drop_last=True,
                pin_memory=True,
                collate_fn=custom_collate(
                    params.data.max_output_dimension,
                    symbol_env.pad_index,
                    params.data.tie_fields,
                    self.params.data.get("mixed_length", 0),
                    params.input_len,
                    params.symbol.pad_right,
                ),
            )
            self.data_iter = iter(self.dataloader)

        self.data_loss = 0.0

        if not params.use_raw_time:
            self.input_len = params.input_len
            self.output_len = params.data.t_num - self.input_len
            if params.rollout:
                self.t = torch.linspace(0, 10, self.input_len + 1, dtype=torch.float32)[None]  # (1, t_num)
            else:
                self.t = torch.linspace(0, 10, params.data.t_num, dtype=torch.float32)[None]  # (1, t_num)

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
            logger.info(f"Found {num:,} parameters in {k}.")
            assert len(v) >= 1

    def set_optimizer(self):
        """
        Set optimizer.
        """
        params = self.params

        if params.optim.type == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.parameters["model"],
                lr=params.optim.lr,
                weight_decay=params.optim.weight_decay,
                eps=params.optim.get("eps", 1e-8),
                amsgrad=params.optim.get("amsgrad", False),
                betas=(0.9, params.optim.get("beta2", 0.999)),
            )
        elif params.optim.type == "adan":
            self.optimizer = DAdaptAdan(
                self.parameters["model"],
                lr=1.0,
                weight_decay=params.optim.weight_decay,
                growth_rate=1.05,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {params.optim.type}")

        if params.optim.scheduler_type:
            if params.optim.scheduler_type == "one_cycle":
                self.scheduler = OneCycleLR(
                    self.optimizer,
                    max_lr=params.optim.lr,
                    div_factor=1e4,
                    pct_start=(params.optim.warmup / params.optim.max_iters),
                    final_div_factor=1e4,
                    steps_per_epoch=params.n_steps_per_epoch,
                    epochs=params.max_epoch,
                )

            else:
                scheduler_args = {}

                if params.optim.scheduler_type == "cosine_with_restarts":
                    scheduler_args["num_cycles"] = params.optim.get("num_cycles", 1)
                elif params.optim.scheduler_type == "cosine_with_min_lr":
                    if "min_lr" in params.optim:
                        scheduler_args["min_lr"] = params.optim.min_lr
                    if "min_lr_rate" in params.optim:
                        scheduler_args["min_lr_rate"] = params.optim.min_lr_rate
                elif params.optim.scheduler_type == "warmup_stable_decay":
                    scheduler_args["num_decay_steps"] = int(params.optim.max_iters * params.optim.decay)
                    scheduler_args["min_lr_ratio"] = params.optim.get("min_lr_ratio", 0)
                    scheduler_args["num_stable_steps"] = (
                        params.optim.max_iters - params.optim.warmup - scheduler_args["num_decay_steps"]
                    )

                self.scheduler = get_scheduler(
                    name=params.optim.scheduler_type,
                    optimizer=self.optimizer,
                    num_warmup_steps=params.optim.warmup,
                    num_training_steps=params.optim.max_iters,
                    scheduler_specific_kwargs=scheduler_args,
                )
        else:
            self.scheduler = None

        logger.info(f"Optimizer: {type(self.optimizer)}, scheduler: {type(self.scheduler)}")

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

        if params.accumulate_gradients > 1:
            loss = loss / params.accumulate_gradients

        # regular optimization
        if not params.amp:
            loss.backward()

            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                if params.clip_grad_norm > 0:
                    clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)
                optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                optimizer.zero_grad()

        # AMP optimization
        else:
            self.scaler.scale(loss).backward()

            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                if params.clip_grad_norm > 0:
                    self.scaler.unscale_(optimizer)
                    clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                optimizer.zero_grad()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % self.params.print_freq != 0:
            return

        # iteration number
        s_iter = "%7i - " % self.n_total_iter

        # learning rates
        s_lr = (" - LR: ") + " / ".join("{:.4e}".format(group["lr"]) for group in self.optimizer.param_groups)

        # memory usage
        max_mem = torch.cuda.max_memory_allocated() / 1024**2
        s_mem = " MEM: {:.2f} MB - ".format(max_mem)

        logger.info(s_iter + s_mem + s_lr)

    def save_checkpoint(self, name, include_optimizer=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return

        path = os.path.join(self.params.dump_path, f"{name}.pth")
        logger.info(f"Saving {name} to {path} ...")

        data = {
            "epoch": self.epoch,
            "n_total_iter": self.n_total_iter,
            "dataloader_count": self.dataloader_count,
            "best_metrics": self.best_metrics,
        }

        for k, v in self.modules.items():
            data[k] = v.state_dict()

        if include_optimizer:
            data["optimizer"] = self.optimizer.state_dict()
            if self.scaler is not None:
                data["scaler"] = self.scaler.state_dict()
            if self.scheduler is not None:
                data["scheduler"] = self.scheduler.state_dict()
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

        if self.params.reload_checkpoint is not None:
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
            try:
                weights = data[k]
                v.load_state_dict(weights)
            except RuntimeError:  # remove the 'module.'
                weights = {name.partition(".")[2]: v for name, v in data[k].items()}
                v.load_state_dict(weights)
            v.requires_grad = requires_grad

        # reload optimizer
        logger.warning("Reloading checkpoint optimizer ...")
        self.optimizer.load_state_dict(data["optimizer"])

        if "scaler" in data and self.scaler is not None:
            logger.warning("Reloading gradient scaler ...")
            self.scaler.load_state_dict(data["scaler"])

        if "scheduler" in data and self.scheduler is not None:
            logger.warning("Reloading scheduler...")
            self.scheduler.load_state_dict(data["scheduler"])

        # reload main metrics
        self.epoch = data["epoch"] + 1
        self.n_total_iter = data["n_total_iter"]
        self.dataloader_count = data["dataloader_count"]
        self.best_metrics = data["best_metrics"]
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

    def end_epoch(self):
        self.save_checkpoint("checkpoint")
        self.epoch += 1

    def get_batch(self):
        """
        Return a training batch
        """
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.dataloader_count += 1
            logger.info(f"Reached end of dataloader, restart {self.dataloader_count}...")
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        return batch

    def data_loss_fn(self, data_output, data_label, data_mask):
        """
        data_output/data_label: Tensor (bs, output_len, x_num, x_num, dim)
        """
        # prepare weights for loss function
        if self.params.loss_weight == "l2":
            weight = torch.linalg.vector_norm(data_label, dim=(2, 3), keepdim=True)  # (bs, output_len, 1, 1, dim)
        elif self.params.loss_weight == "linfty":
            weight, _ = torch.max(torch.abs(data_label), dim=(2, 3), keepdim=True)  # (bs, output_len, 1, 1, dim)
        else:
            weight = None

        if weight is None:
            # no re-weighting, just regular MSE
            loss = F.mse_loss(data_output, data_label, reduction="none")
            loss = loss.sum() / torch.count_nonzero(data_mask.expand_as(loss))
        else:
            # reweight by weight
            eps = 1e-6
            if self.params.square_loss:
                loss = F.mse_loss(data_output, data_label, reduction="none")
                loss = (loss / (weight**2 + eps)).sum() / data_label.size(0)
            else:
                loss = torch.linalg.vector_norm(data_output - data_label, dim=(2, 3), keepdim=True)
                loss = (loss / (weight + eps)).sum() / data_label.size(0)

        return loss

    def normalize_data(self, data_input, data_label=None):
        if self.params.normalize:
            eps = 1e-8
            if self.params.normalize == "meanvar":
                mean = torch.mean(data_input, axis=(1, 2, 3), keepdim=True)  # (bs, 1, 1, 1, dim)
                std = torch.std(data_input, axis=(1, 2, 3), keepdim=True) + eps  # (bs, 1, 1, 1, dim)
            elif self.params.normalize == "range":
                max = torch.amax(data_input, dim=(1, 2, 3), keepdim=True)
                min = torch.amin(data_input, dim=(1, 2, 3), keepdim=True)
                mean = (max + min) / 2
                std = (max - min) / 2 + eps
            elif self.params.normalize == "meanvar_c":
                mean = torch.mean(data_input, axis=(1, 2, 3, 4), keepdim=True)  # (bs, 1, 1, 1, 1)
                std = torch.std(data_input, axis=(1, 2, 3, 4), keepdim=True) + eps  # (bs, 1, 1, 1, 1)
            else:
                raise ValueError(f"Unknown normalization method: {self.params.normalize}")

            data_input = (data_input - mean) / std

            if not self.params.denormalize_for_loss and data_label is not None:
                # compute loss in normalized space
                data_label = (data_label - mean) / std  # use same mean and std

        else:
            mean = 0
            std = 1

        return data_input, data_label, mean, std

    def prepare_data(self, samples, train=True):
        """
        Prepare data for training. (Split entire sequence into input and output, generate loss mask, move to cuda, etc.)

        samples: data:         Tensor     (bs, max_len, x_num, x_num, dim)
                 data_mask:    BoolTensor (bs, 1/output_len, 1, 1, dim)
                 t:            Tensor     (bs, max_len)

        """

        model_input = {}

        data = samples["data"]
        data_mask = samples["data_mask"]  # (bs, 1/output_len, 1, 1, dim)

        if self.params.use_raw_time:
            t = samples["t"]
        else:
            t = self.t

        input_len = self.params.input_len
        data_input = data[:, :input_len]  # (bs, input_len, x_num, x_num, dim)

        # prepare inputs for operator / 1 step training

        data_label = data[:, input_len:]  # (bs, output_len, x_num, x_num, dim)
        data_input, data_label, data_mask = to_cuda(data_input, data_label, data_mask)

        data_input, data_label, mean, std = self.normalize_data(data_input, data_label)

        input_times = t[:, :input_len, None]  # (bs, input_len, 1)
        output_times = (
            t[:, input_len:, None] - input_times[:, -1:]
        )  # (bs, output_len, 1), force a Markovian time stepping

        model_input["input_times"] = to_cuda(input_times)
        model_input["output_times"] = to_cuda(output_times)
        model_input["data_input"] = data_input

        d = {
            "data_label": data_label,
            "data_mask": data_mask,
            "mean": mean,
            "std": std,
        }

        if "symbol_input" in samples:
            model_input["symbol_input"] = to_cuda(samples["symbol_input"])  # LongTensor (bs, symbol_len)
            model_input["symbol_padding_mask"] = to_cuda(samples["symbol_mask"])  # BoolTensor (bs, symbol_len)

        return model_input, d

    def iter(self):
        """
        One training step.
        """
        params = self.params

        samples = self.get_batch()

        model = self.modules["model"]
        model.train()

        # prepare data part

        model_input, d = self.prepare_data(
            samples
        )  # model_input contains model input args, d contains other attributes

        # forward / loss

        """
        Model input: 
            check prepare_data() function

        Model output:
            data_output:  (bs, output_len, x_num, x_num, data_dim)
        """

        with torch.amp.autocast("cpu" if params.cpu else "cuda", enabled=bool(params.amp), dtype=torch.bfloat16):
            data_output = model("fwd", **model_input)  # (bs, output_len, x_num, x_num, data_dim)

            if self.params.normalize and self.params.denormalize_for_loss:
                # denormalize data, compute loss in original space
                data_output = data_output * d["std"] + d["mean"]

            data_output = data_output * d["data_mask"]
            data_loss = self.data_loss_fn(data_output, d["data_label"], d["data_mask"])

        self.data_loss += data_loss.item()

        # optimize
        self.optimize(data_loss)

        self.inner_epoch += 1
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()
