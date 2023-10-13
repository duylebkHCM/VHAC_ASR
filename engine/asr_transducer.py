#!/usr/bin/env/python3
import sys
import torch
import logging
import speechbrain as sb
import sentencepiece
logger = logging.getLogger(__name__)

# Define training procedure
class Conformer_Transducer(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_with_bos, token_with_bos_lens = batch.tokens_bos

        # Add env corruption if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                batch.sig = wavs, wav_lens
                tokens_with_bos = torch.cat(
                    [tokens_with_bos, tokens_with_bos], dim=0
                )
                token_with_bos_lens = torch.cat(
                    [token_with_bos_lens, token_with_bos_lens]
                )
                batch.tokens_bos = tokens_with_bos, token_with_bos_lens

            if hasattr(self.hparams, "speed_perturb"):
                wavs = self.hparams.speed_perturb(wavs)
                
            if hasattr(self.hparams, "augmentation_time"):
                wavs = self.hparams.augmentation_time(wavs, wav_lens)

        # Forward pass
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation_spec") and self.hparams.use_augmentation_spec:
                feats = self.hparams.augmentation_spec(feats)

        # print('feats shape', feats.shape)
        src = self.modules.CNN(feats)
        # print('src', src.shape)
        x = self.modules.enc(src, wav_lens, pad_idx=self.hparams.pad_index)
        x = self.modules.proj_enc(x)

        e_in = self.modules.emb(tokens_with_bos)
        e_in = torch.nn.functional.dropout(
            e_in,
            self.hparams.dec_emb_dropout,
            training=(stage == sb.Stage.TRAIN),
        )
        h, _ = self.modules.dec(e_in)
        h = torch.nn.functional.dropout(
            h, self.hparams.dec_dropout, training=(stage == sb.Stage.TRAIN)
        )
        h = self.modules.proj_dec(h)

        # Joint network
        # add labelseq_dim to the encoder tensor: [B,T,H_enc] => [B,T,1,H_enc]
        # add timeseq_dim to the decoder tensor: [B,U,H_dec] => [B,1,U,H_dec]
        joint = self.modules.Tjoint(x.unsqueeze(2), h.unsqueeze(1))

        # Output layer for transducer log-probabilities
        logits_transducer = self.modules.transducer_lin(joint)

        # Compute outputs
        if stage == sb.Stage.TRAIN:
            p_ctc = None
            p_ce = None

            if (
                self.hparams.ctc_weight > 0.0
                and current_epoch <= self.hparams.number_of_ctc_epochs
            ):
                # Output layer for ctc log-probabilities
                out_ctc = self.modules.proj_ctc(x)
                p_ctc = self.hparams.log_softmax(out_ctc)

            if self.hparams.ce_weight > 0.0:
                # Output layer for ctc log-probabilities
                p_ce = self.modules.dec_lin(h)
                p_ce = self.hparams.log_softmax(p_ce)

            return p_ctc, p_ce, logits_transducer, wav_lens

        elif stage == sb.Stage.VALID:
            hyps, _, _, _ = self.hparams.Greedysearcher(x)            
            return logits_transducer, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (Transducer+(CTC+NLL)) given predictions and targets."""

        ids = batch.id
        tokens, token_lens = batch.tokens
        tokens_eos, token_eos_lens = batch.tokens_eos

        # Train returns 4 elements vs 3 for val and test
        if len(predictions) == 4:
            p_ctc, p_ce, logits_transducer, wav_lens = predictions
        else:
            logits_transducer, wav_lens, predicted_tokens = predictions

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            token_eos_lens = torch.cat([token_eos_lens, token_eos_lens], dim=0)
            tokens = torch.cat([tokens, tokens], dim=0)
            token_lens = torch.cat([token_lens, token_lens], dim=0)

        if stage == sb.Stage.TRAIN:
            CTC_loss = 0.0
            CE_loss = 0.0
            if p_ctc is not None:
                CTC_loss = self.hparams.ctc_cost(
                    p_ctc, tokens, wav_lens, token_lens
                )
            if p_ce is not None:
                CE_loss = self.hparams.ce_cost(
                    p_ce, tokens_eos, length=token_eos_lens
                )
            loss_transducer = self.hparams.transducer_cost(
                logits_transducer, tokens, wav_lens, token_lens
            )
            loss = (
                self.hparams.ctc_weight * CTC_loss
                + self.hparams.ce_weight * CE_loss
                + (1 - (self.hparams.ctc_weight + self.hparams.ce_weight))
                * loss_transducer
            )
        else:
            loss = self.hparams.transducer_cost(
                logits_transducer, tokens, wav_lens, token_lens
            )

        tokenizer = sentencepiece.SentencePieceProcessor(model_file=self.hparams.tokenizer_model)
        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = [
                tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split(" ") for wrd in batch.words]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0

        with self.no_sync(not should_step):
            # Managing automatic mixed precision
            if self.auto_mix_prec:
                with torch.autocast(torch.device(self.device).type):
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)

                # Losses are excluded from mixed precision to avoid instabilities
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

                self.scaler.scale(
                    loss / self.grad_accumulation_factor
                ).backward()

                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    if self.check_gradients(loss):
                        self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.zero_grad(set_to_none=True)
                    self.optimizer_step += 1
                    if isinstance(self.hparams.lr_scheduler, sb.nnet.schedulers.NoamScheduler):
                        self.hparams.lr_scheduler(self.optimizer)
                    elif isinstance(self.hparams.lr_scheduler, sb.nnet.schedulers.WarmCoolDecayLRSchedule):
                        self.hparams.lr_scheduler(self.optimizer, self.optimizer_step)
            else:
                if self.bfloat16_mix_prec:
                    with torch.autocast(
                        device_type=torch.device(self.device).type,
                        dtype=torch.bfloat16,
                    ):
                        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                        loss = self.compute_objectives(
                            outputs, batch, sb.Stage.TRAIN
                        )
                else:
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                    loss = self.compute_objectives(
                        outputs, batch, sb.Stage.TRAIN
                    )
                (loss / self.grad_accumulation_factor).backward()
                if should_step:
                    if self.check_gradients(loss):
                        self.optimizer.step()
                    self.zero_grad(set_to_none=True)
                    self.optimizer_step += 1
                    if isinstance(self.hparams.lr_scheduler, sb.nnet.schedulers.NoamScheduler):
                        self.hparams.lr_scheduler(self.optimizer)
                    elif isinstance(self.hparams.lr_scheduler, sb.nnet.schedulers.WarmCoolDecayLRSchedule):
                        self.hparams.lr_scheduler(self.optimizer, self.optimizer_step)
                    
        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            if isinstance(self.hparams.lr_scheduler, sb.nnet.schedulers.NewBobScheduler):
                old_lr_model, new_lr_model = self.hparams.lr_scheduler(
                    stage_stats["loss"]
                )
                sb.nnet.schedulers.update_learning_rate(
                    self.optimizer, new_lr_model
                )
                lr = old_lr_model
            else:
                if hasattr(self.hparams.lr_scheduler, 'current_lr'):
                    lr = self.hparams.lr_scheduler.current_lr
                else:
                    lr = self.optimizer.param_groups[0]["lr"]
                
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }

            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            # We save multiple checkpoints as we will average them!
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"], "epoch": epoch},
                min_keys=["WER"],
            )

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()
