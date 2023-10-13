#!/usr/bin/env/python3
import sys
import logging
import torch
import speechbrain as sb
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
sys.path.append('.')
from engine import ASR_Controller
from data_builder import dataio_prepare
import argparse

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    
    config = args.config
    
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments([config])
    asr_engine = Path(config).parent.stem.lower()
    
    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # We can now directly create the datasets for training, valid, and test
    data_process_obj = dataio_prepare
    results = data_process_obj(hparams)

    # Trainer initialization
    asr_brain = getattr(ASR_Controller, asr_engine)(
        modules=hparams["modules"],
        opt_class=hparams.get("opt_class", None),
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    if results.get('label_encoder', None) is not None:
        # We dynamicaly add the tokenizer to our brain class.
        # NB: This tokenizer corresponds to the one used for the LM!!
        asr_brain.tokenizer = results['label_encoder']
    
    train_dataloader_opts = hparams.get("train_dataloader_opts", None) or results.get("train_loader_kwargs", None)
    valid_dataloader_opts = hparams.get("valid_dataloader_opts", None) or results.get("valid_loader_kwargs", None)

    if results.get('train_bsampler', None) is not None:
        collate_fn = None
        if "collate_fn" in train_dataloader_opts:
            collate_fn = train_dataloader_opts["collate_fn"]

        train_dataloader_opts = {
            "batch_sampler": results['train_bsampler'],
            "num_workers": hparams["num_workers"],
            "pin_memory": True
        }

        if collate_fn is not None:
            train_dataloader_opts["collate_fn"] = collate_fn

    if results.get('valid_bsampler', None) is not None:
        collate_fn = None
        if "collate_fn" in valid_dataloader_opts:
            collate_fn = valid_dataloader_opts["collate_fn"]

        valid_dataloader_opts = {
            "batch_sampler": results['valid_bsampler'], 
            "pin_memory": True,  
            "num_workers": hparams["num_workers"]
        }

        if collate_fn is not None:
            valid_dataloader_opts["collate_fn"] = collate_fn
    
    with torch.autograd.detect_anomaly():
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            results['train_data'],
            results["valid_data"],
            train_loader_kwargs=train_dataloader_opts,
            valid_loader_kwargs=valid_dataloader_opts,
        )

    # Save final checkpoint (fixed name)
    asr_brain.checkpointer.save_checkpoint(name="latest")
