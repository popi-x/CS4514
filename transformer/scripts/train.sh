accelerate launch --config_file config/train.yaml train.py \
--train_metadata differ_list.json \
--valid_metadata val_list.json \
--svg_folder ../stroke/tokenized_file/ \
--img_folder ../pytorch-vqvae/img_tokens/ \
--output_dir proj_log/ \
--project_name Sketch_naive \
--maxlen 1666 \
--batchsize 8
