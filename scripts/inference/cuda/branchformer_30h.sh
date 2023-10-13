CUDA_VISIBLE_DEVICES=0 python inference.py \
--model_path weights/TRANSDUCER_BPE_viettel_asr_30h/210901/CKPT+2023-09-21+22-19-45+00 \
--config_path branchformer_inference.yaml \
--test_path dataset/raw_data/private_test/wav/ \
--submission_path prediction_output/BRANCHFORMER_TRANSDUCER/ \
--start_idx 0 \
--end_idx 11 \
--batch_size 96 \
--device cuda