GPU=0
user="jingmins"

directory="dataset/"
ICs_equation=50


datasets=(  burgers conservation_sinflux conservation_cubicflux inviscid_burgers inviscid_conservation_sinflux inviscid_conservation_cubicflux advection diff_bistablereact_1D fplanck heat  Klein_Gordon diff_linearreact_1D diff_squarelogisticreact_1D cahnhilliard_1D Sine_Gordon kdv diff_logisreact_1D wave)

for dataset in "${datasets[@]}"; do
    # Cleanup old data


    rm -r checkpoint/${user}/dumped/${dataset}/gen_data
    mkdir -p ${directory}/${dataset}/
     # Generating data
    CUDA_VISIBLE_DEVICES=$GPU python3 train.py --num_workers 12 --export_data --exp_name ${dataset} --exp_id gen_data --max_epoch 1 --n_steps_per_epoch 500 --log_periodic 1 --ICs_per_equation $ICs_equation --t_range 2.0 --t_num 64 --x_num 128 --ode_param_range_gamma 0.1 --types pde_${dataset} --max_output_dim 1 --batch_size 256

    # Copy data to a new location
    filename=${dataset}_${ICs_equation}
    mv checkpoint/${user}/dumped/${dataset}/gen_data/text.prefix ${directory}/${dataset}/${filename}.prefix
    mv checkpoint/${user}/dumped/${dataset}/gen_data/data.h5 ${directory}/${dataset}/${filename}_data.h5



done
