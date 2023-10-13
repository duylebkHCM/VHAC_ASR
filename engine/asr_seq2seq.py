#!/usr/bin/env/python3
import torch
import logging
import sentencepiece
import speechbrain as sb
import torch
from torch.nn import ModuleList

logger = logging.getLogger(__name__)


# Brain class for speech recognition training
class CRDNN(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""
    def __init__(self, modules=None, opt_class=None, hparams=None, run_opts=None, checkpointer=None, profiler=None):
        super().__init__(modules, opt_class, hparams, run_opts, checkpointer, profiler)
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=hparams['tokenizer_model'])

    def compute_forward(self, batch, stage):
        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        feats, self.feat_lens = self.prepare_features(stage, batch.sig)
        tokens_bos, _ = self.prepare_tokens(stage, batch.tokens_bos)
        
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation_spec") and self.hparams.use_augmentation_spec:
                feats = self.hparams.augmentation_spec(feats)

        # Running the encoder (prevent propagation to feature extraction)
        if isinstance(self.modules.encoder, ModuleList):
            encoded_signal = feats.detach()
            for m in self.modules.encoder:
                encoded_signal = m(encoded_signal)
        else:
            encoded_signal = self.modules.encoder(feats.detach())

        # Embed tokens and pass tokens & encoded signal to decoder
        embedded_tokens = self.modules.embedding(tokens_bos)
        decoder_outputs, _ = self.modules.decoder(
            embedded_tokens, encoded_signal, self.feat_lens
        )

        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(decoder_outputs)
        predictions = {"seq_logprobs": self.hparams.log_softmax(logits)}

        if self.is_ctc_active(stage):
            # Output layer for ctc log-probabilities
            ctc_logits = self.modules.ctc_lin(encoded_signal)
            predictions["ctc_logprobs"] = self.hparams.log_softmax(ctc_logits)
        
        if stage == sb.Stage.VALID:
            predictions["tokens"], _ = self.hparams.valid_search(
                encoded_signal, self.feat_lens
            )

        return predictions

    def is_ctc_active(self, stage):
        """Check if CTC is currently active.

        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        """
        if stage != sb.Stage.TRAIN:
            return False
        current_epoch = self.hparams.epoch_counter.current
        return current_epoch <= self.hparams.number_of_ctc_epochs

    def prepare_features(self, stage, wavs):
        """Prepare features for computation on-the-fly

        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        wavs : tuple
            The input signals (tensor) and their lengths (tensor).
        """
        wavs, wav_lens = wavs

        # Add augmentation if specified. In this version of augmentation, we
        # concatenate the original and the augment batches in a single bigger
        # batch. This is more memory-demanding, but helps to improve the
        # performance. Change it if you run OOM.
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])

            if hasattr(self.hparams, "speed_perturb"):
                wavs = self.hparams.speed_perturb(wavs)
                
            if hasattr(self.hparams, "augmentation_time"):
                wavs = self.hparams.augmentation_time(wavs, wav_lens)

        # Feature computation and normalization
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        return feats, wav_lens

    def prepare_tokens(self, stage, tokens):
        """Double the tokens batch if features are doubled.

        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        tokens : tuple
            The tokens (tensor) and their lengths (tensor).
        """
        tokens, token_lens = tokens
        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens = torch.cat([tokens, tokens], dim=0)
            token_lens = torch.cat([token_lens, token_lens], dim=0)
        return tokens, token_lens

    def compute_objectives(self, predictions, batch, stage):
        # Compute sequence loss against targets with EOS
        tokens_eos, tokens_eos_lens = self.prepare_tokens(
            stage, batch.tokens_eos
        )
        loss = sb.nnet.losses.nll_loss(
            log_probabilities=predictions["seq_logprobs"],
            targets=tokens_eos,
            length=tokens_eos_lens,
            label_smoothing=self.hparams.label_smoothing,
        )

        # Add ctc loss if necessary. The total cost is a weighted sum of
        # ctc loss + seq2seq loss
        if self.is_ctc_active(stage):
            # Load tokens without EOS as CTC targets
            tokens, tokens_lens = self.prepare_tokens(stage, batch.tokens)
            loss_ctc = self.hparams.ctc_cost(
                predictions["ctc_logprobs"], tokens, self.feat_lens, tokens_lens
            )
            loss *= 1 - self.hparams.ctc_weight
            loss += self.hparams.ctc_weight * loss_ctc

        if stage != sb.Stage.TRAIN:
            # Converted predicted tokens from indexes to words
            predicted_words = [
                self.tokenizer.decode_ids(prediction).split(" ")
                for prediction in predictions["tokens"]
            ]
            target_words = [words.split(" ") for words in batch.words]

            # Monitor word error rate and character error rated at
            # valid and test time.
            self.wer_metric.append(batch.id, predicted_words, target_words)
            self.cer_metric.append(batch.id, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        # Set up statistics trackers for this stage
        # In this case, we would like to keep track of the word error rate (wer)
        # and the character error rate (cer)
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        # Store the train loss until the validation stage.
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Summarize the statistics from the stage for record-keeping.
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.test_wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        if self.hparams.average_ckpt:
            ckpts = self.checkpointer.find_checkpoints(
                max_key=max_key, min_key=min_key
            )
            ckpt = sb.utils.checkpoints.average_checkpoints(
                ckpts, recoverable_name="model", device=self.device
            )

            self.hparams.model.load_state_dict(ckpt, strict=True)
            self.hparams.model.eval()
