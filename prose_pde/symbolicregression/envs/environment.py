from logging import getLogger
import os
import io
import copy
import json
from collections import defaultdict

# import math
import numpy as np
import symbolicregression.envs.generators as generators
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .utils import *
from ..utils import bool_flag
import h5py

SPECIAL_WORDS = [
    "<BOS>",
    "<EOS>",
    # "<X>",
    # "</X>",
    # "<Y>",
    # "</Y>",
    # "</POINTS>",
    # "<INPUT_PAD>",
    # "<OUTPUT_PAD>",
    "<PAD>",
    "<PLACEHOLDER>",
    # "(",
    # ")",
    # "SPECIAL",
    # "OOD_unary_op",
    # "OOD_binary_op",
    # "OOD_constant",
]
logger = getLogger()

SKIP_ITEM = "SKIP_ITEM"


class FunctionEnvironment(object):
    TRAINING_TASKS = {"functions"}

    def __init__(self, params):
        self.params = params
        self.rng = None
        self.seed = None
        self.float_precision = params.float_precision
        self.mantissa_len = params.mantissa_len
        self.max_size = None
        self.output_dim = params.max_output_dimension
        self.float_tolerance = 10 ** (-params.float_precision)
        self.additional_tolerance = [10 ** (-i) for i in range(params.float_precision + 1)]
        assert (params.float_precision + 1) % params.mantissa_len == 0, "Bad precision/mantissa len ratio"

        self.generator = generators.RandomFunctions(params, SPECIAL_WORDS)
        self.float_encoder = self.generator.float_encoder
        self.float_words = self.generator.float_words
        self.equation_encoder = self.generator.equation_encoder
        self.equation_words = self.generator.equation_words
        self.equation_words += self.float_words

        # number of words / indices
        self.float_id2word = {i: s for i, s in enumerate(self.float_words)}
        self.equation_id2word = {i: s for i, s in enumerate(self.equation_words)}
        self.float_word2id = {s: i for i, s in self.float_id2word.items()}
        self.equation_word2id = {s: i for i, s in self.equation_id2word.items()}

        assert len(self.float_words) == len(set(self.float_words))
        assert len(self.equation_word2id) == len(set(self.equation_word2id))
        self.n_words = params.n_words = len(self.equation_words)
        logger.info(f"vocabulary: {len(self.float_word2id)} float words, {len(self.equation_word2id)} equation words")

    def batch_data_operator(self, data, t_eval, step=1, dims=None):
        """
        Batch list of data into a Tensor ready for operator data decoder
        Outputs:
            data_input   (input_len, bs, output_dim+1)
            data_label   (data_len, bs, output_dim)
            lengths      (bs, )
            dims         (bs, )
        """
        length = t_eval.size(0)
        input_len = self.params.input_len // step
        lengths = torch.LongTensor([input_len for _ in data])
        if dims is None:
            dims = torch.LongTensor([eq.size(-1) for eq in data])
        else:
            dims = torch.LongTensor(dims)

        data_input = torch.zeros(input_len, len(data), self.output_dim + 1, dtype=t_eval.dtype)
        data_label = torch.zeros(length, len(data), self.output_dim, dtype=t_eval.dtype)

        t_input = t_eval[0 : self.params.input_len : step]

        for i, eq in enumerate(data):
            data_input[:, i, 0].copy_(t_input)
            data_input[:, i, 1 : (dims[i] + 1)].copy_(eq[0 : self.params.input_len : step, : dims[i]])
            data_label[:, i, : dims[i]].copy_(eq[..., dims[i]])

        return data_input, data_label, lengths, dims

    def get_tx_grid(self, t_grid, x_grid, space_dim, x_grid_size):
        """
        Generate tx_grid based on spacial and temporal grids
        Inputs:
            t_grid       (t_num, )
            x_grid       (x_num, ..., x_num, space_dim)
        Outputs:
            tx_grid      (t_num*x_grid_size, space_dim + 1)
        """

        t_num = t_grid.size(0)

        x_grid = torch.reshape(x_grid, (1, x_grid_size, space_dim))
        t_grid = torch.reshape(t_grid, (t_num, 1))

        tx_grid = torch.zeros(t_num, x_grid_size, space_dim + 1, dtype=t_grid.dtype)

        tx_grid[:, :, 0] = t_grid.expand(t_num, x_grid_size)
        tx_grid[:, :, 1:] = x_grid.expand(t_num, x_grid_size, space_dim)

        return torch.reshape(tx_grid, (t_num * x_grid_size, space_dim + 1))

    def batch_data_pde_operator(self, data, t_eval, x_grid_size, step=1, dims=None):
        """
        Batch list of data into a Tensor ready for operator data decoder
        Inputs:
            data           (bs, t_num, x_grid_size, dim)
        Outputs:
            data_input     (input_len, bs, 1+output_dim*x_grid_size)
            data_label     (data_len=t_num, bs, x_grid_size, output_dim)
            lengths        (bs, )
            dims           (bs, )
        """
        length = t_eval.size(0)
        input_len = self.params.input_len // step
        lengths = torch.LongTensor([input_len for _ in data])
        if dims is None:
            dims = torch.LongTensor([eq.size(-1) for eq in data])
        else:
            dims = torch.LongTensor(dims)

        if self.params.separate_modality:
            # data_input1 = torch.zeros(input_len, len(data), 1 + self.output_dim * x_grid_size, dtype=t_eval.dtype)
            # data_label1 = torch.zeros(length, len(data), x_grid_size, self.output_dim, dtype=t_eval.dtype)
            # t_input = t_eval[0 : self.params.input_len : step]

            # for i, eq in enumerate(data):
            #     # logger.info(f"{i}: eq shape {eq.shape}, input_len {input_len}, dim {dims[i]}")
            #     data_input1[:, i, 0].copy_(t_input)
            #     data_input1[:, i, 1 : (dims[i] * x_grid_size + 1)].copy_(
            #         eq[0 : self.params.input_len : step, :, : dims[i]].reshape(input_len, dims[i] * x_grid_size)
            #     )

            #     data_label1[:, i, :, : dims[i]].copy_(eq[..., : dims[i]])

            data_label = torch.stack(data).transpose(0, 1)  # (data_len, bs, x_grid_size, output_dim)

            reshape_dim = x_grid_size * data_label.size(-1)
            bs = len(dims)
            t_input = t_eval[0 : self.params.input_len : step][:, None, None]  # (data_len, 1, 1)
            t_input = t_input.expand(input_len, bs, 1)
            data_input = data_label[0 : self.params.input_len : step].reshape(input_len, bs, reshape_dim)
            data_input = torch.cat((t_input, data_input), dim=-1)

            # assert torch.allclose(data_label1, data_label)
            # assert torch.allclose(data_input1, data_input)

        else:
            data_input = torch.zeros(input_len, len(data), 1 + self.output_dim * x_grid_size, dtype=t_eval.dtype)
            data_label = torch.zeros(length, len(data), x_grid_size, self.output_dim, dtype=t_eval.dtype)

            t_input = t_eval[0 : self.params.input_len : step]

            for i, eq in enumerate(data):
                # logger.info(f"{i}: eq shape {eq.shape}, input_len {input_len}, dim {dims[i]}")
                data_input[:, i, 0].copy_(t_input)
                data_input[:, i, 1 : (dims[i] * x_grid_size + 1)].copy_(
                    eq[0 : self.params.input_len : step, :, :].reshape(input_len, dims[i] * x_grid_size)
                )

                data_label[:, i, :, : dims[i]].copy_(eq)

        return data_input, data_label, lengths, dims

    def batch_data_window(self, data, t_eval, window_size):
        """
        Batch list of data into a Tensor ready for sliding window data decoder
        Outputs:
            data_input   (input_len, bs, output_dim+1)
            data_label   (window_size+input_len, bs, output_dim+1)
        """
        length = t_eval.size(0)
        assert length % 2 == 0
        lengths = torch.LongTensor([len(eq) // 2 for eq in data])
        input_len = length // 2

        data_input = torch.zeros(input_len, len(data), self.output_dim + 1, dtype=t_eval.dtype)
        data_label = torch.zeros(length - input_len + window_size, len(data), self.output_dim + 1, dtype=t_eval.dtype)

        for i, eq in enumerate(data):
            data_input[:, i, 0].copy_(t_eval[0:input_len])
            data_input[:, i, 1:].copy_(eq[0:input_len])
            data_label[:, i, 0].copy_(t_eval[input_len - window_size :])
            data_label[:, i, 1:].copy_(eq[input_len - window_size :])

        return data_input, data_label, lengths

    def batch_data_old(self, data, t_eval):
        """
        Take as input a list of n sequences (torch.Tensor vectors) and
        time stamps. Return a tensor of size (slen, n, 2) where slen
        is the length of the longest sentence, and a vector lengths containing
        the length of each sentence.
        """
        assert t_eval.size(0) == len(data[0])
        lengths = torch.LongTensor([len(eq) for eq in data])
        assert lengths.min().item() == lengths.max().item()
        sent = torch.zeros(lengths.max().item(), lengths.size(0), self.output_dim + 1, dtype=t_eval.dtype)

        for i, eq in enumerate(data):
            sent[0 : lengths[i], i, 0].copy_(t_eval)
            sent[0 : lengths[i], i, 1].copy_(eq)
        return sent, lengths

    def batch_equations(self, equations):
        """
        Batch list of text seq into a Tensor ready for text decoder
        Input:
            equations  list of n sequences (torch.LongTensor vectors)
        Outputs:
            sent       Tensor (slen, n) where slen is the length of the longest sentence
            lengths    (slen, ) length of each sentence
        """
        lengths = torch.LongTensor([2 + len(eq) for eq in equations])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.float_word2id["<PAD>"])
        sent[0] = self.equation_word2id["<BOS>"]
        for i, eq in enumerate(equations):
            sent[1 : lengths[i] - 1, i].copy_(eq)
            sent[lengths[i] - 1, i] = self.equation_word2id["<EOS>"]
        return sent, lengths

    def batch_equations_placeholder(self, equations):
        """
        Batch list of text seq into a Tensor ready for text encoder (no <BOS> or <EOS>)
        Input:
            equations  list of n sequences (torch.LongTensor vectors)
        Outputs:
            sent       Tensor (slen, n) where slen is the length of the longest sentence
            lengths    (slen, ) length of each sentence
        """
        lengths = torch.LongTensor([len(eq) for eq in equations])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.float_word2id["<PAD>"])
        for i, eq in enumerate(equations):
            sent[: lengths[i], i].copy_(eq)
        return sent, lengths

    def word_to_idx(self, words, float_input=True):
        if float_input:
            return [[torch.LongTensor([self.float_word2id[dim] for dim in point]) for point in seq] for seq in words]
        else:
            return [torch.LongTensor([self.equation_word2id[w] for w in eq]) for eq in words]

    def word_to_infix(self, words, is_float=True, str_array=True):
        if is_float:
            m = self.float_encoder.decode(words)
            if m is None:
                return None
            if str_array:
                return np.array2string(np.array(m))
            else:
                return np.array(m)
        else:
            m = self.equation_encoder.decode(words)
            if m is None:
                return None
            if str_array:
                return m.infix()
            else:
                return m

    def idx_to_infix(self, lst, is_float=True, str_array=True):
        if is_float:
            idx_to_words = [self.float_id2word[int(i)] for i in lst]
        else:
            idx_to_words = [self.equation_id2word[int(term)] for term in lst]
        return self.word_to_infix(idx_to_words, is_float, str_array)

    def gen_expr(self, train):
        errors = defaultdict(int)
        while True:
            try:
                expr, error = self._gen_expr(train)
                if error:
                    errors[error[0]] += 1
                    assert False
                return expr, errors
            except:
                if self.params.debugging:
                    print(error)
                continue

    def _gen_expr(self, train):
        item = self.generator.generate_one_sample(self.rng, train=train)

        tree = item["tree"]

        if len(item["data"]) == 0:
            return item, ["data generation error"]

        if "tree_encoded" not in item:
            tree_encoded = self.equation_encoder.encode(tree)
            assert all([x in self.equation_word2id for x in tree_encoded]), "tree: {}\n encoded: {}".format(
                tree, tree_encoded
            )

            item["tree_encoded"] = tree_encoded

        return item, []

    def preprocess(self, x, train):
        # cur_data = torch.FloatTensor(x["data"])
        # del x["data"]
        # x["data"] = cur_data
        # if not train:
        #     x["tree"] = self.equation_encoder.decode(x["tree_encoded"])
        return x

    def preload_data(self, params, path, train):
        if path is None:
            return None

        assert os.path.isfile(path), "{} not found".format(path)
        logger.info(f"Preloading data from {path} ...")
        with io.open(path, mode="r", encoding="utf-8") as f:
            # either reload the entire file, or the first N lines
            # (for the training set)
            if not train:
                lines = []
                for i, line in enumerate(f):
                    if i % params.n_gpu_per_node == params.local_rank:
                        lines.append(self.preprocess(json.loads(line.rstrip()), train))
            else:
                lines = []
                for i, line in enumerate(f):
                    if i == params.reload_size:
                        break
                    if i % params.n_gpu_per_node == params.local_rank:
                        lines.append(self.preprocess(json.loads(line.rstrip()), train))

        data = lines

        logger.info(f"Loaded {len(data)} equations from the disk.")
        return data

    def create_train_iterator(self, task, data_path, params, **args):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")
        # data = self.preload_data(params, None if data_path is None else data_path[task][0], train=True)
        dataset = EnvDataset(
            self,
            task,
            train=True,
            skip=False,
            params=params,
            path=(None if data_path is None else data_path[task][0]),
            # data=data,
            **args,
        )

        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 3600),
            batch_size=params.batch_size,
            num_workers=(params.num_workers if data_path is None or params.num_workers == 0 else 1),
            shuffle=data_path is not None,
            collate_fn=dataset.collate_fn,
            drop_last=True,
            # pin_memory=True,
        )

    def create_test_iterator(
        self,
        data_type,
        task,
        data_path,
        batch_size,
        params,
        size,
        **args,
    ):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating {data_type} iterator for {task} ...")
        # data = self.preload_data(
        #     params, None if data_path is None else data_path[task][int(data_type[5:])], train=False
        # )
        dataset = EnvDataset(
            self,
            task,
            train=False,
            skip=False,
            params=params,
            path=(None if data_path is None else data_path[task][int(data_type[5:])]),
            size=size,
            type=data_type,
            # data=data,
            **args,
        )

        return DataLoader(
            dataset,
            timeout=0,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            # pin_memory=True,
        )

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument(
            "--use_sympy",
            type=bool_flag,
            default=False,
            help="Whether to use sympy parsing (basic simplification)",
        )
        parser.add_argument(
            "--use_abs",
            type=bool_flag,
            default=False,
            help="Whether to replace log and sqrt by log(abs) and sqrt(abs)",
        )

        # generator

        parser.add_argument(
            "--types",
            type=str,
            default="pde",
            # default="chaotic_ode_all",
            help="types of ODE/PDE examples to generate, select from chaotic_ode_all, chaotic_ode_3d, pde, pde_mol",
        )
        # parser.add_argument(
        #     "--eval_types",
        #     type=str,
        #     default="",
        #     # default="chaotic_ode_all",
        #     help="types of ODE/PDE examples to generate, select from chaotic_ode_all, chaotic_ode_3d, pde, pde_mol",
        # )

        # parser.add_argument(
        #     "--types_PDE",
        #     type=str,
        #     default="pde",
        #     help="types of PDE examples to generate",
        # )

        # parser.add_argument(
        #     "--ode_gen",
        #     type=bool_flag,
        #     default=True,
        #     help="Whether to generate ODEs",
        # )
        # parser.add_argument(
        #     "--pde_gen",
        #     type=bool_flag,
        #     default=True,
        #     help="Whether to generate PDEs",
        # )
        parser.add_argument(
            "--ICs_per_equation",
            type=int,
            default=20,
            help="maximum number of initial conditions to sample for each equation",
        )
        parser.add_argument(
            "--IC_types",
            type=str,
            default="train",
            help="Type of initial conditions of PDEs, selected from train,eval",
        )
        parser.add_argument("--IC_range", type=float, default=1.0, help="IC limit for PDE")
        parser.add_argument("--IC_jump_start", type=float, default=0, help="For extrapolation study, start point of IC")
        parser.add_argument("--IC_jump_end", type=float, default=2, help="For extrapolation study, start point of IC")
        parser.add_argument("--use_sharedIC", action="store_true", default=False, help="Use shared initial conditions")
        parser.add_argument("--use_sharedcoeff", action="store_true", default=False,
                            help="Use shared coefficients")
        parser.add_argument("--t_range", type=float, default=6.0, help="Time upper limit for ODE/PDE that rescale to (0,t_range)")
        parser.add_argument("--t_final",type=str, default="", help="Time upper limit each specific PDE, typea,valuea;typeb,valueb")
        parser.add_argument("--t_num", type=int, default=32, help="Number of evaluation points for ODE")
        parser.add_argument("--x_range", type=float, default=2.0, help="Spacial grid range (for each dimension)")
        parser.add_argument("--x_num", type=int, default=64, help="Spacial grid size (for each dimension)")
        parser.add_argument(
            "--x_patch_size",
            type=int,
            default=64,
            help="size of space grid to use for single element in input sequence",
        )
        parser.add_argument("--input_len", type=int, default=16, help="Number of input data points")
        parser.add_argument("--input_step", type=int, default=1, help="Step size for input data points")
        parser.add_argument("--train_output_step", type=int, default=1, help="Step size for input data points")
        parser.add_argument("--train_output_start", type=int, default=16, help="starting point of training output")
        parser.add_argument("--eval_output_step", type=int, default=1, help="Step size for input data points")

        parser.add_argument(
            "--eval_output_start",
            type=int,
            default=16,
            help="starting point of evaluation output",
        )
        parser.add_argument(
            "--ode_param_range_gamma",
            type=float,
            default=0.1,
            help="relative range for sampling parameters of ODEs",
        )

        parser.add_argument("--max_pde_spatialdim", type=float, default=1, help="maximum spatial dim for PDEs")

        parser.add_argument(
            "--max_int",
            type=int,
            default=20,
            help="Maximal integer in symbolic expressions",
        )

        parser.add_argument(
            "--float_precision",
            type=int,
            default=2,
            help="Number of digits in the mantissa",
        )
        parser.add_argument(
            "--mantissa_len",
            type=int,
            default=1,
            help="Number of tokens for the mantissa (must be a divisor of float_precision+1)",
        )
        parser.add_argument("--max_exponent", type=int, default=10, help="Maximal order of magnitude")
        parser.add_argument(
            "--max_exponent_prefactor",
            type=int,
            default=1,
            help="Maximal order of magnitude in prefactors",
        )
        parser.add_argument("--min_input_dimension", type=int, default=1)
        parser.add_argument("--max_input_dimension", type=int, default=1)
        parser.add_argument("--min_output_dimension", type=int, default=1)
        # parser.add_argument("--max_output_dimension", type=int, default=3)
        parser.add_argument("--max_output_dimension", type=int, default=1)


class EnvDataset(Dataset):
    def __init__(
        self,
        env,
        task,
        train,
        params,
        path,
        skip=False,
        size=None,
        type=None,
        data=None,
        **args,
    ):
        super(EnvDataset).__init__()
        self.env = env
        self.train = train
        self.skip = skip
        self.task = task
        self.batch_size = params.batch_size
        self.env_base_seed = params.env_base_seed
        self.path = path
        self.count = 0
        self.remaining_data = 0
        self.type = type
        self.params = params
        self.errors = defaultdict(int)
        self.tree_skeletons = dict()

        if "test_env_seed" in args:
            self.test_env_seed = args["test_env_seed"]
        else:
            self.test_env_seed = None
        if "env_info" in args:
            self.env_info = args["env_info"]
        else:
            self.env_info = None

        assert task in FunctionEnvironment.TRAINING_TASKS
        assert size is None or not self.train
        assert not params.batch_load or params.reload_size > 0
        # batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size

        self.batch_load = params.batch_load
        self.reload_size = params.reload_size
        self.local_rank = params.local_rank

        self.basepos = 0
        self.nextpos = 0
        self.seekpos = 0

        # generation, or reloading from file
        if path is not None:
            assert os.path.isfile(path), "{} not found".format(path)
            if data is not None:
                self.data = data
            elif params.batch_load and self.train:
                self.load_chunk()
            else:
                logger.info(f"Loading data from {path} ...")
                with io.open(path, mode="r", encoding="utf-8") as f:
                    # either reload the entire file, or the first N lines
                    # (for the training set)
                    if not train:
                        lines = []
                        for i, line in enumerate(f):
                            # if i % 1000 == 999:
                            #     print(i)
                            if i % params.n_gpu_per_node == params.local_rank:
                                lines.append(self.preprocess(json.loads(line.rstrip())))
                    else:
                        lines = []
                        for i, line in enumerate(f):
                            # if i % 10000 == 9999:
                            #     print(i)
                            if i == params.reload_size:
                                break
                            if i % params.n_gpu_per_node == params.local_rank:
                                lines.append(self.preprocess(json.loads(line.rstrip())))

                self.data = lines

                if self.params.separate_modality:
                    filename = path[:-7] + "_data.h5"
                    assert os.path.isfile(filename), "Data file {} not found".format(path)
                    with h5py.File(filename, "r") as hf:
                        self.data_matrix = hf["data"][params.local_rank :: params.n_gpu_per_node]

                    assert "dim" in self.data[0]
                    assert len(self.data_matrix) == len(self.data), "Dataset size mismatch"

                    logger.info(f"Data size: {self.data_matrix.shape}.")

                logger.info(f"Loaded {len(self.data)} equations from the disk.")

        # dataset size: default 512000 iterator for train, 25600 for valid / test

        if params.dry_run or params.eval_only:
            if size is not None and size > 0:
                self.size = size
                assert size <= len(self.data)
            elif path is not None:
                self.size = len(self.data)
            elif self.train:
                self.size = 512000
            else:
                self.size = 25600
        else:
            if path is not None:
                self.size = len(self.data)
            elif size is not None and size > 0:
                self.size = size
                assert size <= len(self.data)
            elif self.train:
                self.size = 512000
            else:
                self.size = 25600

        self.modes = None
        if params.noisy_text_input and not train:
            # pregenerate text noise so test set is the same across different runs
            self.init_rng()
            add_prob = params.add_term_prob
            miss_prob = params.miss_term_prob
            self.modes = self.env.rng.choice(
                np.array([0, 1, -1]),
                size=self.size,
                p=np.array([1 - add_prob - miss_prob, add_prob, miss_prob]),
            )

            self.tree_skeletons = []

            for i in range(self.size):
                mode = self.modes[i]
                if str(self.params.types).startswith("pde"):
                    tree_skeleton = self.env.generator.pde_generator.get_skeleton_tree(
                        self.data[i]["type"], mode=mode, rng=self.env.rng
                    )
                else:
                    tree_skeleton = self.env.generator.ode_generator.get_skeleton_tree(
                        self.data[i]["type"], mode=mode, rng=self.env.rng
                    )
                self.tree_skeletons.append(tree_skeleton)

    def preprocess(self, x):
        # cur_data = torch.FloatTensor(x["data"])
        # del x["data"]
        # x["data"] = cur_data
        # if not self.train:
        #     x["tree"] = self.env.equation_encoder.decode(x["tree_encoded"])
        return x

    def load_chunk(self):
        assert not self.params.separate_modality

        self.basepos = self.nextpos
        logger.info(f"Loading data from {self.path} ... seekpos {self.seekpos}, " f"basepos {self.basepos}")
        endfile = False
        with io.open(self.path, mode="r", encoding="utf-8") as f:
            f.seek(self.seekpos, 0)
            lines = []
            for i in range(self.reload_size):
                line = f.readline()
                if not line:
                    endfile = True
                    break
                if i % self.params.n_gpu_per_node == self.local_rank:
                    lines.append(line.rstrip().split("|"))
            self.seekpos = 0 if endfile else f.tell()

        self.data = [xy.split("\t") for _, xy in lines]
        self.data = [xy for xy in self.data if len(xy) == 2]
        self.nextpos = self.basepos + len(self.data)
        logger.info(
            f"Loaded {len(self.data)} equations from the disk. seekpos {self.seekpos}, " f"nextpos {self.nextpos}"
        )
        if len(self.data) == 0:
            self.load_chunk()

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        samples = zip_dic(elements)
        errors = copy.deepcopy(self.errors)
        self.errors = defaultdict(int)
        return samples, errors

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if self.env.rng is not None:
            return
        if self.train:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            seed = [worker_id, self.params.global_rank, self.env_base_seed]
            if self.env_info is not None:
                seed += [self.env_info]
            self.env.rng = np.random.RandomState(seed)
            self.env.seed = seed
            logger.info(
                f"Initialized random generator for worker {worker_id}, with seed "
                f"{seed} "
                f"(base seed={self.env_base_seed})."
            )
        else:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            seed = [
                worker_id,
                self.params.global_rank,
                self.test_env_seed if "valid" in self.type else 0,
            ]
            self.env.rng = np.random.RandomState(seed)
            self.env.seed = seed
            logger.info(
                "Initialized {} generator, with seed {} (random state: {})".format(self.type, seed, self.env.rng)
            )

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0), "issue in worker id"
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    def __getitem__(self, index):
        """
        Return a training sample.
        Either generate it, or read it from file.
        """
        self.init_rng()
        if self.path is None:
            if self.train and self.skip:
                return SKIP_ITEM
            else:
                sample = self.generate_sample(index)

        else:
            if self.train and self.skip:
                return SKIP_ITEM
            else:
                sample = self.read_sample(index)

        return sample

    def read_sample(self, index):
        """
        Read a sample.
        """
        idx = index
        if self.train and self.batch_load:
            if index >= self.nextpos:
                self.load_chunk()
            idx = index - self.basepos

        # x = copy.deepcopy(self.data[idx])

        # x["data"] = torch.FloatTensor(x["data"])

        cur_data = self.data[idx]
        x = dict()

        if self.params.separate_modality:
            x["data"] = torch.from_numpy(self.data_matrix[idx]).float()
        else:
            x["data"] = torch.FloatTensor(cur_data["data"])

        for key in cur_data.keys():
            if key != "data":
                x[key] = copy.deepcopy(cur_data[key])

        if not self.train:
            x["tree"] = self.env.equation_encoder.decode(x["tree_encoded"])

        if self.params.use_skeleton:
            if self.params.noisy_text_input:
                if self.train:
                    # generate text noise on the fly
                    add_prob = self.params.add_term_prob
                    miss_prob = self.params.miss_term_prob
                    mode = self.env.rng.choice(
                        np.array([0, 1, -1]),
                        p=np.array([1 - add_prob - miss_prob, add_prob, miss_prob]),
                    )
                    if str(self.params.types).startswith("pde"):
                        x["tree_skeleton"] = self.env.generator.pde_generator.get_skeleton_tree(
                            x["type"], mode=mode, rng=self.env.rng
                        )
                    else:
                        x["tree_skeleton"] = self.env.generator.ode_generator.get_skeleton_tree(
                            x["type"], mode=mode, rng=self.env.rng
                        )
                    x["mode"] = mode
                else:
                    # for testing, use pregenerated ones
                    x["mode"] = self.modes[index]
                    x["tree_skeleton"] = self.tree_skeletons[index]
            else:
                if str(self.params.types).startswith("pde"):
                    x["tree_skeleton"] = self.env.generator.pde_generator.get_skeleton_tree(
                        x["type"], mode=0, rng=self.env.rng
                    )
                else:
                    x["tree_skeleton"] = self.env.generator.ode_generator.get_skeleton_tree(
                        x["type"], mode=0, rng=self.env.rng
                    )
        return x

    def generate_sample(self, index=None):
        """
        Generate a sample.
        """
        if self.remaining_data == 0:
            self.item, errors = self.env.gen_expr(self.train)
            for error, count in errors.items():
                self.errors[error] += count

            self.remaining_data = len(self.item["data"])

        self.remaining_data -= 1
        data = self.item["data"][-self.remaining_data]

        sample = dict()
        sample["type"] = self.item["type"]
        sample["tree"] = self.item["tree"]
        sample["data"] = data

        sample["tree_encoded"] = self.item["tree_encoded"]

        if self.params.use_skeleton:
            if self.params.noisy_text_input:
                if self.modes is not None and index is not None:
                    mode = self.modes[index]
                else:
                    add_prob = self.params.add_term_prob
                    miss_prob = self.params.miss_term_prob
                    mode = self.env.rng.choice(
                        np.array([0, 1, -1]),
                        p=np.array([1 - add_prob - miss_prob, add_prob, miss_prob]),
                    )
                if str(self.params.types).startswith("pde"):
                    sample["tree_skeleton"] = self.env.generator.pde_generator.get_skeleton_tree(
                        sample["type"], mode=mode, rng=self.env.rng
                    )
                else:
                    sample["tree_skeleton"] = self.env.generator.ode_generator.get_skeleton_tree(
                        sample["type"], mode=mode, rng=self.env.rng
                    )
                sample["mode"] = mode
            else:
                if str(self.params.types).startswith("pde"):
                    sample["tree_skeleton"] = self.env.generator.pde_generator.get_skeleton_tree(
                        sample["type"], mode=0, rng=self.env.rng
                    )
                else:
                    sample["tree_skeleton"] = self.env.generator.ode_generator.get_skeleton_tree(
                        sample["type"], mode=0, rng=self.env.rng
                    )

        self.count += 1
        return sample


def select_dico_index(dico, idx):
    new_dico = {}
    for k in dico.keys():
        new_dico[k] = dico[k][idx]
    return new_dico
