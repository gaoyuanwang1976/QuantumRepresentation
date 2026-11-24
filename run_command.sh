
conda activate qvae

python QVAE.py -o cobyla -e 5 -p 2 -t 5 -i dataset/dim64_01234.dat --initial_point 0 --import_path dataset/ --input_dim 64 --beta_weight 0.2 --num_layers 3 --reconstruction_loss fidelity --regularizer_type JSD --input_task mnist --circuit_type real_amp --decoder_inv --cost_type latent --output_dir "raw01234_dim1_output/ini0_beta0.2"
