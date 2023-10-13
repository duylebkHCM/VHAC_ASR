python viettel_asr_prepare.py \
--build_option 2 \
--pseudo_data_path dataset/raw_data/dataset2/training_100h/wavs/ \
--real_data_path dataset/raw_data/dataset1/training_30h/wavs/ \
--pseudo_label_csv dataset/pseudo_label/v1/kenlm_postprocess.csv \
--save_json_train dataset/processed_label/combine_label/v1/train.json \
--save_json_valid dataset/processed_label/combine_label/v1/valid.json