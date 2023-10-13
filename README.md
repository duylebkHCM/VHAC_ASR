# VHAC ASR CHALLENGE
## Team Name: Speechless

## Inference process

- Step 1: Clone this repo 
- Step 2: Download checkpoint from https://drive.google.com/file/d/1OZ8REeoKk-Kv_OZ1QhOVzDNiXjYrWupN/view?usp=drive_link, and put it in the VHAC_ASR folder and unzip checkpoint
- Step 3: Put the private dataset in `dataset/raw_data` folder
- Step 4: Pull docker image from docker hub 
```
docker pull duybktiengiang1603/vhac_asr:latest
```
- Step 5: Run docker container and mount the folder to docker container
```bash
docker run --gpus all -v $PWD/VHAC_ASR:/src/ --name vhac --rm -dt duybktiengiang1603/vhac_asr:latest
```
- Step 6: Run to open terminal in docker container
```
docker exec -it vhac /bin/bash
```

- Step 7: Activate conda 
```
conda activate duyla_seqmodel && cd /src/
```

- Step 8: Run bash script to execute inference code. Here we use 4 models for inference 

```
bash scripts/inference/cuda/branchformer_30h.sh
bash scripts/inference/cuda/conformer_30h.sh
bash scripts/inference/cuda/large_branchformer_30h100h.sh
bash scripts/inference/cuda/large_conformer_30h100h.sh
```

- Step 9: Perform ensemble on 4 model results 

```
cd ensemble/
python combine_result.py
```

The final result is in `ensemble/combine_result` 


## Training process

- Step 1: Clone this repo 
- Step 2: Put dataset1.zip and dataset2.zip dataset in `dataset/raw_data` and unzip.
- Step 3: Pull docker image from docker hub 
```
docker pull duybktiengiang1603/vhac_asr:latest
```

**Our proposed method also use kenlm as a statistical language model which is training on label data of 30h and 100h dataset for postprocess the result of pseudo labels. We use the kenlm score to select between the prediction of model or the given labels of 100h dataset. Here is the training step.**
- Step 3.5:
```
cd LM && bash install.sh && bash train.sh
```

- Step 4: Run docker container and mount the folder to docker container
```bash
docker run --gpus all -v $PWD/VHAC_ASR:/src/ --name vhac --rm -dt duybktiengiang1603/vhac_asr:latest
```
- Step 5: Run to open terminal in docker container
```
docker exec -it vhac /bin/bash
```

- Step 6: Activate conda 
```
conda activate duyla_seqmodel && cd /src/
```

Below is step-by-step guide to perform our proposed method

First, build label for training30h dataset
- Step 1:
```
bash scripts/data_processing/create_30h_label.sh
```

Build tokenizer
- Step 2:
```
bash scripts/tokenizer/build_tokenizer_1k.sh
```

Start training model Conformer on 30h dataset
- Step 3:
```
bash scripts/training/conformer_30h.sh
```

Training model Branchformer on 30h dataset
- Step 4:
```
bash scripts/training/branchformer_30h.sh
```

Next, use Conformer on 30h dataset model to create pseudo label on 100h dataset
- Step 5:
```
bash scripts/pseudo_label_generate/build_pseudo_v1.sh
```

Next, postprocess pseudo labels
- Step 6:
```
bash scripts/data_processing/process_pseudo_label_v1.sh
```

Combine pseudo label with 30h dataset label
- Step 7:
```
bash scripts/data_processing/create_100h30h_label_v1.sh
```

Start training third model - Conformer on 30h_100h dataset
- Step 8:
```
bash scripts/training/large_conformer_30h100h.sh
```

Build another tokenizer with 2K token
- Step 9:
```
bash scripts/tokenizer/build_tokenizer_2k.sh
```

Use Conformer on 30h_100h dataset model to generate more pseudo label 
- Step 10:
```
bash scripts/pseudo_label_generate/build_pseudo_v2.sh
```

Next, postprocess pseudo labels
- Step 11:
```
bash scripts/data_processing/process_pseudo_label_v2.sh
```

Combine pseudo label with 30h dataset label
- Step 12:
```
bash scripts/data_processing/create_100h30h_label_v2.sh
```

Start training large version of branchformer model
- Step 13:
```
bash scripts/training/large_branchformer_30h100h.sh
```