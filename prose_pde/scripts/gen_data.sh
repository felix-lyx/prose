GPU=0
user="jingmins"

directory="dataset"
dataset="test_1"


 rm -r checkpoint/${user}/dumped/${dataset}/gen_val_data
 rm -r checkpoint/${user}/dumped/${dataset}/gen_train_data
 rm -r checkpoint/${user}/dumped/${dataset}/gen_test_data
 mkdir  ${directory}/${dataset}/
 #
 CUDA_VISIBLE_DEVICES=$GPU python3 train.py --export_data --exp_name ${dataset} --exp_id gen_val_data --max_epoch 1 --n_steps_per_epoch 100 --log_periodic 1 --ICs_per_equation 4 --t_range 2.0 --t_num 64 --x_num 128 --ode_param_range_gamma 0.1 --types pde --max_output_dim 1 --batch_size 256  &&

 filename="val_25600"
 cp checkpoint/${user}/dumped/${dataset}/gen_val_data/text.prefix ${directory}/${dataset}/${filename}.prefix &&
 cp checkpoint/${user}/dumped/${dataset}/gen_val_data/data.h5 ${directory}/${dataset}/${filename}_data.h5 &&


 CUDA_VISIBLE_DEVICES=$GPU python3 train.py --export_data --exp_name ${dataset} --exp_id gen_train_data --max_epoch 1 --n_steps_per_epoch 2000 --log_periodic 1 --ICs_per_equation 20 --t_range 2.0 --t_num 64 --x_num 128 --ode_param_range_gamma 0.1 --types pde --max_output_dim 1 --batch_size 256&&
 filename="train_512000"
 cp checkpoint/${user}/dumped/${dataset}/gen_train_data/text.prefix ${directory}/${dataset}/${filename}.prefix &&
 cp checkpoint/${user}/dumped/${dataset}/gen_train_data/data.h5 ${directory}/${dataset}/${filename}_data.h5 &&
 #
 CUDA_VISIBLE_DEVICES=$GPU python3 train.py --export_data --exp_name ${dataset} --exp_id gen_test_data --max_epoch 1 --n_steps_per_epoch 400 --log_periodic 1 --ICs_per_equation 4 --t_range 2.0 --t_num 64 --x_num 128  --ode_param_range_gamma 0.1 --types pde --max_output_dim 1 --batch_size 256 &&
 filename="test_102400"
 cp checkpoint/${user}/dumped/${dataset}/gen_test_data/text.prefix ${directory}/${dataset}/${filename}.prefix &&
 cp checkpoint/${user}/dumped/${dataset}/gen_test_data/data.h5 ${directory}/${dataset}/${filename}_data.h5 &&

# #