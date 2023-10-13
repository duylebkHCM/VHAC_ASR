CUDA_VISIBLE_DEVICES=0 python inference.py \
--model_path weights/TRANSDUCER_BPE_viettel_asr_100h_30h_more_data/CKPT+2023-10-10+09-20-43+00 \
--config_path conformer_inference.yaml \
--test_path dataset/raw_data/private_test/wav/ \
--submission_path prediction_output/BRANCHFORMER_TRANSDUCER_30H100H_moredata/ \
--start_idx 0 \
--end_idx 5 \
--batch_size 96 \
--device cuda