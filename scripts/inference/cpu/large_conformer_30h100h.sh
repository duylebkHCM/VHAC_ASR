CUDA_VISIBLE_DEVICES='' python inference.py \
--model_path weights/TRANSDUCER_BPE_viettel_asr_100h_30h/CKPT+2023-10-03+14-39-56+00 \
--config_path conformer_inference.yaml \
--test_path dataset/raw_data/private_test/wav/ \
--submission_path prediction_output/CONFORMER_TRANSDUCER_30H100H/ \
--start_idx -1 \
--end_idx 20 \
--batch_size 64 \
--device cuda