CUDA_VISIBLE_DEVICES=0 python inference.py \
--model_path weights/TRANSDUCER_BPE_viettel_asr_100h_30h/CKPT+2023-10-03+14-39-56+00 \
--config_path conformer_inference.yaml \
--test_path dataset/raw_data/dataset2/training100h/wavs/ \
--submission_path dataset/pseudo_label/v2 \
--start_idx 0 \
--end_idx 600 \
--batch_size 64 \
--device cuda