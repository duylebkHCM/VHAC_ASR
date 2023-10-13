CUDA_VISIBLE_DEVICES='' python inference.py \
--model_path weights/TRANSDUCER_BPE_viettel_asr_30h/190901/CKPT+2023-09-22+01-11-41+00 \
--config_path conformer_transducer_inference.yaml \
--test_path dataset/raw_data/private_test/wav/ \
--submission_path prediction_output/CONFORMER_TRANSDUCER/ \
--start_idx -1 \
--end_idx -1 \
--batch_size 48 \
--device cuda