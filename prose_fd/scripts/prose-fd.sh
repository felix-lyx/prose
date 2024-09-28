GPU=0
GPUs=0,1

# main model in paper

expid=prose_fd
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py exp_id=${expid} batch_size=88 data=fluids_sample optim=wsd model=prose_2to1 &&
CUDA_VISIBLE_DEVICES=$GPU python main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=256 model=prose_2to1

# updated main model with better performance (main difference is rmsnorm instead of layernorm)

expid=prose_fd_rms
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py dryrun=1 exp_id=${expid} batch_size=80 data=fluids_sample optim=wsd model=prose_2to1_rms compile=1 &&
CUDA_VISIBLE_DEVICES=$GPU python main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=256 model=prose_2to1_rms


### Ablation Studies

# no symbolic information, i.e., 1to1 model

expid=prose_fd_1to1
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py exp_id=${expid} batch_size=64 data=fluids_sample optim=wsd symbol.symbol_input=0 model=prose_1to1 &&
CUDA_VISIBLE_DEVICES=$GPU python main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=256 symbol.symbol_input=0 model=prose_1to1 exp_name=debug_eval

# rollout in time instead of operator

expid=prose_fd_rollout
CUDA_VISIBLE_DEVICES=$GPU python main.py exp_id=${expid} batch_size=176 data=fluids_sample optim=wsd model=prose_2to1 data.t_num=11 &&
CUDA_VISIBLE_DEVICES=$GPU python main.py eval_only=1 rollout=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=256 model=prose_2to1