import argparse
from symbolicregression.envs import ENVS
from symbolicregression.utils import bool_flag


def get_parser():
    """
    Generate a parameters parser.
    """

    # parse parameters
    parser = argparse.ArgumentParser(description="PROSE", add_help=False)

    ########################### important ones ###############################

    # some default values, can also be changed in command line

    hidden_dim = 512  # default 512
    n_head = 8

    n_text_enc_layers = 4  # default 4
    n_text_dec_layers = 8  # default 8
    n_data_enc_layers = 2  # default 2
    n_data_dec_layers = 8  # default 8
    n_fusion_layers = 8  # default 8

    num_workers = 8  # only for generating dataset
    batch_size = 256  # default 224/256, training batch size
    batch_size_eval = 256  # default 256
    eval_size = 25600  # eval dataset size

    dataset_path = (
        "functions,../dataset/ode_3d/train_512000.prefix,../dataset/ode_3d/val_25600.prefix,"
    )

    # main parameters

    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="enable dry_run (1 epoch 5 steps)",
    )
    parser.add_argument("--cpu", action="store_true", default=False, help="Run on CPU")
    parser.add_argument("--eval_in_domain", type=bool_flag, default=True)
    parser.add_argument(
        "--use_wandb",
        type=bool_flag,
        default=True,
        help="Use wandb to record experiments",
    )
    parser.add_argument(
        "--save_periodic",
        type=int,
        default=20,  # 25
        help="Save the model periodically (0 to disable)",
    )
    parser.add_argument(
        "--log_periodic",
        type=int,
        default=50,
        help="Log stats periodically (0 to disable)",
    )
    parser.add_argument("--text_only", action="store_true", default=False, help="no data decoder")
    parser.add_argument("--data_only", action="store_true", default=False, help="no text decoder")
    parser.add_argument("--no_text", action="store_true", default=False, help="no text at all")
    parser.add_argument(
        "--split_fused_feature_text",
        default=True,
        type=bool_flag,
        help="whether to split fused features for text decoder",
    )
    parser.add_argument(
        "--split_fused_feature_data",
        default=True,
        type=bool_flag,
        help="whether to split fused features for data decoder",
    )
    parser.add_argument(
        "--data_feature_resnet",
        type=bool_flag,
        default=False,
        help="use one-layer ResNet on fused feature for data operator",
    )
    parser.add_argument(
        "--data_decoder_attn",
        type=bool_flag,
        default=False,
        help="add self-attention on query locations in data decoder part",
    )
    parser.add_argument(
        "--use_skeleton",
        type=bool_flag,
        default=True,
        help="use a skeleton rather than functions with constants",
    )
    parser.add_argument(
        "--noisy_text_input",
        type=bool_flag,
        default=True,
        help="randomly add/delete terms in text input",
    )
    parser.add_argument(
        "--add_term_prob",
        type=float,
        default=0.15,
        help="probability of randomly adding a term",
    )
    parser.add_argument(
        "--miss_term_prob",
        type=float,
        default=0.15,
        help="probability of randomly deleting a term",
    )
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb experiment name")
    parser.add_argument("--wandb_id", type=str, default=None, help="Wandb run id")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=num_workers,
        help="Number of CPU workers for DataLoader",
    )
    parser.add_argument("--exp_name", type=str, default="ode_exp", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--print_freq", type=int, default=500, help="Print every n steps")

    # training parameters

    parser.add_argument("--max_epoch", type=int, default=80, help="Number of epochs")
    parser.add_argument(
        "--n_steps_per_epoch",
        type=int,
        default=2000,
        help="Number of steps per epoch",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        # default='adam_cosine,warmup_updates=1000',
        default="adam_inverse_sqrt,warmup_updates=20000,weight_decay=0.0001",
        help="Optimizer (SGD / RMSprop / Adam, etc.)",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=batch_size,
        help="Number of sentences per batch",
    )
    parser.add_argument(
        "--data_loss_weight",
        type=float,
        default=6.0,
        help="weight for data loss",
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=1.0,
        help="Clip gradients norm (0 to disable)",
    )
    parser.add_argument(
        "--train_noise_gamma",
        type=float,
        default=0.02,
        help="Should we train with additional output noise",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="additive",
        choices=["additive", "multiplicative"],
        help="Type of noise added",
    )
    parser.add_argument(
        "--stopping_criterion",
        type=str,
        default="",
        help="Stopping criterion, and number of non-increase before stopping the experiment",
    )
    parser.add_argument(
        "--amp",
        type=int,
        default=0,
        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.",
    )
    parser.add_argument(
        "--eval_amp",
        type=int,
        default=0,
        help="Use AMP wrapper during evaluation. -1 to disable.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="use torch.compile to speed up",
    )

    # saving and loading data

    parser.add_argument("--dump_path", type=str, default="", help="Experiment dump path")
    parser.add_argument("--eval_dump_path", type=str, default=None, help="Evaluation dump path")
    parser.add_argument(
        "--save_results",
        type=bool_flag,
        default=False,
        help="Should we save results?",
    )
    parser.add_argument(
        "--export_data",
        default=False,
        action="store_true",
        help="Export data and disable training.",
    )
    parser.add_argument(
        "--reload_data",
        type=str,
        default=dataset_path,
        help="Load dataset from the disk (task1,train_path1,valid_path1,test_path1;task2,train_path2,valid_path2,test_path2)",
    )
    parser.add_argument(
        "--reload_size",
        type=int,
        default=-1,
        help="Reloaded training set size (-1 for everything)",
    )
    parser.add_argument(
        "--batch_load",
        type=bool_flag,
        default=False,
        help="Load training set by batches (of size reload_size).",
    )

    # environment parameters
    parser.add_argument("--env_name", type=str, default="functions", help="Environment name")
    ENVS[parser.parse_known_args("")[0].env_name].register_args(parser)

    # tasks
    parser.add_argument("--tasks", type=str, default="functions", help="Tasks")

    # beam search configuration

    parser.add_argument(
        "--beam_eval",
        type=bool_flag,
        default=True,
        help="Evaluate with beam search decoding.",
    )
    parser.add_argument(
        "--max_generated_output_len",
        type=int,
        default=150,
        help="Max generated output length",
    )
    parser.add_argument(
        "--beam_eval_train",
        type=int,
        default=0,
        help="At training time, number of validation equations to test the model on using beam search (-1 for everything, 0 to disable)",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam size, default = 1 (greedy decoding)",
    )
    parser.add_argument(
        "--beam_type",
        type=str,
        default="sampling",
        help="Beam search or sampling",
    )
    parser.add_argument(
        "--beam_temperature",
        type=int,
        default=0.1,
        help="Beam temperature for sampling",
    )
    parser.add_argument(
        "--beam_length_penalty",
        type=float,
        default=1,
        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.",
    )
    parser.add_argument(
        "--beam_early_stopping",
        type=bool_flag,
        default=True,
        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.",
    )
    parser.add_argument("--beam_selection_metrics", type=int, default=1)
    parser.add_argument("--max_number_bags", type=int, default=1)

    # reload pretrained model / checkpoint
    parser.add_argument("--reload_model", type=str, default="", help="Reload a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="", help="Reload a checkpoint")

    # evaluation
    parser.add_argument(
        "--batch_size_eval",
        type=int,
        default=batch_size_eval,
        help="Number of sentences per batch during evaluation (if None, set to 1.5*batch_size)",
    )
    parser.add_argument(
        "--validation_metrics",
        type=str,
        default="_data_loss",
        help="What metrics should we report? accuracy_tolerance/_l1_error/r2/_complexity/_relative_complexity/is_symbolic_solution",
    )
    parser.add_argument(
        "--debug_train_statistics",
        type=bool_flag,
        default=False,
        help="whether we should print infos distributions",
    )
    parser.add_argument(
        "--eval_noise_gamma",
        type=float,
        default=0.02,
        help="Should we train with additional output noise",
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=eval_size,
        help="Size of valid and test samples",
    )
    parser.add_argument(
        "--eval_only", action="store_true", default=False, help="Only run evaluations"
    )
    parser.add_argument("--print_outputs", action="store_true", default=False, help="print outputs")
    parser.add_argument(
        "--text_ode_solve", action="store_true", default=False, help="use text output as ODE map"
    )
    parser.add_argument("--eval_from_exp", type=str, default="", help="Path of experiment to use")
    parser.add_argument("--eval_data", type=str, default="", help="Path of data to eval")

    ################## unimportant ones, no need to change #################

    parser.add_argument("--n_trees_to_refine", type=int, default=10, help="refine top n trees")

    # float16 / AMP API
    parser.add_argument("--fp16", type=bool_flag, default=False, help="Run model with float16")
    parser.add_argument(
        "--rescale",
        type=bool_flag,
        default=True,
        help="Whether to rescale at inference.",
    )
    parser.add_argument(
        "--env_base_seed",
        type=int,
        default=-1,
        help="Base seed for environments (-1 to use timestamp seed)",
    )
    parser.add_argument("--test_env_seed", type=int, default=42, help="Test seed for environments")
    parser.add_argument(
        "--accumulate_gradients",
        type=int,
        default=1,
        help="Accumulate model gradients over N iterations (N times larger batch sizes)",
    )

    # evaluation

    parser.add_argument(
        "--refinements_types",
        type=str,
        default="method=BFGS_batchsize=256_metric=/_mse",
        help="What refinement to use. Should separate by _ each arg and value by =. None does not do any refinement",
    )

    parser.add_argument("--eval_verbose", type=int, default=0, help="Export evaluation details")
    parser.add_argument(
        "--eval_verbose_print",
        type=bool_flag,
        default=False,
        help="Print evaluation details",
    )

    # debug
    parser.add_argument(
        "--debug_slurm",
        type=bool_flag,
        default=False,
        help="Debug multi-GPU / multi-node within a SLURM job",
    )
    parser.add_argument("--debug", help="Enable all debug flags", action="store_true")

    # CPU / multi-gpu / multi-node

    parser.add_argument("--local-rank", type=int, default=-1, help="Multi-GPU - Local rank")
    parser.add_argument(
        "--master_port",
        type=int,
        default=-1,
        help="Master port (for multi-node SLURM jobs)",
    )
    parser.add_argument(
        "--windows",
        type=bool_flag,
        default=False,
        help="Windows version (no multiprocessing for eval)",
    )
    parser.add_argument(
        "--nvidia_apex", type=bool_flag, default=False, help="NVIDIA version of apex"
    )

    # model parameters

    # text

    parser.add_argument(
        "--text_enc_emb_dim",
        type=int,
        default=hidden_dim,
        help="Text Encoder embedding layer size",
    )
    parser.add_argument(
        "--text_dec_emb_dim",
        type=int,
        default=hidden_dim,
        help="Text Decoder embedding layer size",
    )
    parser.add_argument(
        "--n_text_enc_layers",
        type=int,
        default=n_text_enc_layers,
        help="Number of Transformer layers in the text encoder",
    )
    parser.add_argument(
        "--n_text_dec_layers",
        type=int,
        default=n_text_dec_layers,
        help="Number of Transformer layers in the text decoder",
    )
    parser.add_argument(
        "--n_text_enc_heads",
        type=int,
        default=n_head,
        help="Number of Transformer text encoder heads",
    )
    parser.add_argument(
        "--n_text_dec_heads",
        type=int,
        default=n_head,
        help="Number of Transformer text decoder heads",
    )
    parser.add_argument(
        "--n_text_enc_hidden_layers",
        type=int,
        default=1,
        help="Number of FFN layers in Transformer text encoder",
    )
    parser.add_argument(
        "--n_text_dec_hidden_layers",
        type=int,
        default=1,
        help="Number of FFN layers in Transformer text decoder",
    )
    parser.add_argument(
        "--text_enc_positional_embeddings",
        type=str,
        default="sinusoidal",
        help="Use none/learnable/sinusoidal/alibi embeddings for text",
    )
    parser.add_argument(
        "--text_dec_positional_embeddings",
        type=str,
        default="sinusoidal",
        help="Use none/learnable/sinusoidal/alibi embeddings for text",
    )
    parser.add_argument(
        "--text_share_inout_emb",
        type=bool_flag,
        default=True,
        help="Share input and output embeddings",
    )

    # data

    parser.add_argument(
        "--data_enc_emb_dim",
        type=int,
        default=hidden_dim,
        help="Data Encoder embedding layer size",
    )
    parser.add_argument(
        "--data_dec_emb_dim",
        type=int,
        default=hidden_dim,
        help="Data Decoder embedding layer size",
    )
    parser.add_argument(
        "--n_data_enc_layers",
        type=int,
        default=n_data_enc_layers,
        help="Number of Transformer layers in the data encoder",
    )
    parser.add_argument(
        "--n_data_dec_layers",
        type=int,
        default=n_data_dec_layers,
        help="Number of Transformer layers in the data decoder",
    )
    parser.add_argument(
        "--n_data_enc_heads",
        type=int,
        default=n_head,
        help="Number of Transformer data encoder heads",
    )
    parser.add_argument(
        "--n_data_dec_heads",
        type=int,
        default=n_head,
        help="Number of Transformer data decoder heads",
    )
    parser.add_argument(
        "--n_data_enc_hidden_layers",
        type=int,
        default=1,
        help="Number of FFN layers in Transformer data encoder",
    )
    parser.add_argument(
        "--n_data_dec_hidden_layers",
        type=int,
        default=1,
        help="Number of FFN layers in Transformer data decoder",
    )
    parser.add_argument(
        "--data_enc_positional_embeddings",
        type=str,
        default=None,
        help="Use none/learnable/sinusoidal/alibi embeddings for data",
    )
    parser.add_argument(
        "--data_dec_positional_embeddings",
        type=str,
        default=None,
        help="Use none/learnable/sinusoidal/alibi embeddings for data",
    )

    # Fusion

    parser.add_argument(
        "--fusion_emb_dim",
        type=int,
        default=hidden_dim,
        help="Fusion embedding layer size",
    )
    parser.add_argument(
        "--n_fusion_layers",
        type=int,
        default=n_fusion_layers,
        help="Number of Transformer layers in fusion",
    )
    parser.add_argument(
        "--n_fusion_heads",
        type=int,
        default=n_head,
        help="Number of fusion Transformer heads",
    )
    parser.add_argument(
        "--n_fusion_hidden_layers",
        type=int,
        default=1,
        help="Number of FFN layers in fusion Transformer",
    )
    parser.add_argument(
        "--fusion_positional_embeddings",
        type=str,
        default=None,
        help="Use none/learnable/sinusoidal/alibi embeddings for fusion",
    )
    parser.add_argument(
        "--fusion_type_embeddings",
        type=bool_flag,
        default=True,
        help="Add an additional type embedding for different modality",
    )

    # general parameters for Transformers

    parser.add_argument(
        "--use_library_attention",
        type=bool_flag,
        default=True,
        help="Use library attention or self-implementation",
    )
    parser.add_argument(
        "--norm_attention",
        type=bool_flag,
        default=False,
        help="Normalize attention and train temperature in Transformer",
    )
    parser.add_argument("--dropout", type=float, default=0, help="Dropout")
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0,
        help="Dropout in the attention layer",
    )

    return parser
