GPU=0
GPUs=0,1

# FNO

expid=fno
CUDA_VISIBLE_DEVICES=$GPU python main.py exp_id=${expid} batch_size=64 symbol.symbol_input=0 model=fno amp=0 max_epoch=20 &&
CUDA_VISIBLE_DEVICES=$GPU python main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=128 symbol.symbol_input=0 model=fno amp=0

# UNet

expid=unet_1
CUDA_VISIBLE_DEVICES=$GPU python main.py exp_id=${expid} batch_size=160 symbol.symbol_input=0 model=unet &&
CUDA_VISIBLE_DEVICES=$GPU python main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=256 symbol.symbol_input=0 model=unet

# ViT

expid=vit
CUDA_VISIBLE_DEVICES=$GPU python main.py exp_id=${expid} batch_size=96 symbol.symbol_input=0 model=vit &&
CUDA_VISIBLE_DEVICES=$GPU python main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=128 symbol.symbol_input=0 model=vit

# DeepONet

expid=don
CUDA_VISIBLE_DEVICES=$GPU python main.py exp_id=${expid} batch_size=128 symbol.symbol_input=0 model=deeponet &&
CUDA_VISIBLE_DEVICES=$GPU python main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=128 symbol.symbol_input=0 model=deeponet