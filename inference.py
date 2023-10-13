from speechbrain.pretrained import EncoderDecoderASR, EncoderASR
import sys
from pathlib import Path
import pandas as pd
import torch
import speechbrain
from speechbrain.nnet.containers import LengthsCapableSequential
from speechbrain.dataio.batch import PaddedBatch
import editdistance
import argparse


def compute_accuracy(gt_text, pred_text):
    words_gt = gt_text.split()
    words_pred = pred_text.split()

    cur_max_len = max(len(words_gt), len(words_pred))
    max_char_len = max(len(gt_text), len(pred_text))

    norm_ED = 1 - editdistance.eval(pred_text, gt_text) / max_char_len
    word_NED = 1 - editdistance.eval(words_gt, words_pred) / cur_max_len
    is_correct = int(gt_text == pred_text)

    return norm_ED, word_NED, is_correct


class CustomEncoderDecoderASR(EncoderDecoderASR):
    def __init__(self, modules=None, hparams=None, run_opts=None, freeze_params=True):
        super().__init__(modules=modules, hparams=hparams, run_opts=run_opts, freeze_params=freeze_params)
        # Put modules on the right device, accessible with dot notation
        self.mods = torch.nn.ModuleDict(modules)
        for name, module in self.mods.items():
            if module is not None and name == 'encoder':
                module.to(self.device)
                if isinstance(module, LengthsCapableSequential):
                    for sub_name, sub_module in module.items():
                        sub_module.to(self.device)
            if module is not None and name == "decoder":
                module.to(self.device)
                if isinstance(module, speechbrain.decoders.transducer.TransducerBeamSearcher):
                    for item in module.classifier_network:
                        item.to(self.device)
                    for item in module.decode_network_lst:
                        item.to(self.device)
                        
    def encode_batch(self, wavs, wav_lens):
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        encoder_out = self.mods.encoder(wavs, wav_lens)
        return encoder_out

    def transcribe_batch(self, wavs, wav_lens, get_conf):
        
        with torch.no_grad():
            wav_lens: torch.Tensor = wav_lens.to(self.device)
            wavs: torch.Tensor = wavs.to(self.device)
            
            encoder_out = self.encode_batch(wavs, wav_lens)
            if isinstance(self.mods.decoder, speechbrain.decoders.transducer.TransducerBeamSearcher):
                predicted_tokens, _, _, log_probs = self.mods.decoder(encoder_out)
            else:
                predicted_tokens, _ = self.mods.decoder(encoder_out, wav_lens)
                
            predicted_words = [
                self.tokenizer.decode_ids(token_seq)
                for token_seq in predicted_tokens
            ]
        
        if get_conf:
            return predicted_words, predicted_tokens, log_probs
        
        return predicted_words, predicted_tokens

    def transcribe_batch_file(self, paths, get_conf, **kwargs):
        wavforms = []
        for path in paths:
            wav = self.load_audio(path=path.as_posix())
            assert wav.ndim == 1
            wavforms.append({"wav":wav})
            
        batch = PaddedBatch(wavforms, padded_keys="wav", padding_kwargs={"value":self.hparams.pad_index})
        
        if get_conf:
            predicted_words, _, log_probs = self.transcribe_batch(
                batch.wav.data, batch.wav.lengths, get_conf
            )
        
            return predicted_words, log_probs
        else:
            predicted_words, _ = self.transcribe_batch(
                batch.wav.data, batch.wav.lengths, get_conf
            )
        
            return predicted_words
    

def build_indices_plain(list_images, batch_size):
    lst_indices = list(range(len(list_images)))
    batch_lists = [lst_indices[i:i + batch_size] for i in range(0, len(lst_indices), batch_size)]
    return batch_lists


def update_df(path, save_df, is_check_lbl, total_correct, pred_transcript, log_prob=None):
    if is_check_lbl:
        lbl = info.get(path, None)
        if lbl is not None:
            _, word_NED, is_correct = compute_accuracy(lbl.lower(), pred_transcript)
            save_df['wer'].append(word_NED)
            
            if is_correct:
                total_correct += 1
                save_df['iscorrect'].append(1)
            else:
                save_df['iscorrect'].append(0)
        else:
            save_df['wer'].append(1.0)
            save_df['iscorrect'].append(0)

    save_df['id'].append(path)
    save_df['transcript'].append(pred_transcript)
    
    if log_prob: 
        save_df['conf'].append(log_prob.item())
    else:
        save_df['conf'].append(0.0)
        
    return total_correct


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Path to model checkpoint')
    parser.add_argument('--config_path', help='Name of config file for inference')
    parser.add_argument('--test_path', help='Path to wav file for infer')
    parser.add_argument('--submission_path', help='Path to result file or pseudo label file')
    parser.add_argument('--start_idx', type=int, default=0, help='Batch idx for start run inference')
    parser.add_argument('--end_idx', type=int, default=-1, help='End batch idx for infer')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size for infer')
    parser.add_argument('--device', default='cuda', help='Device to run')
    args = parser.parse_args()
    
    model_path=args.model_path
    config_path=args.config_path
    test_path=args.test_path
    submission_path=args.submission_path

    batch_size = args.batch_size
    get_conf = True
    start_idx = args.start_idx
    end_idx = args.end_idx
        
    if not Path(submission_path).exists():
        Path(submission_path).mkdir(parents=True)

    test_list = sorted(list(Path(test_path).rglob("*.wav")))
    check_lbl = False
    total_correct = 0
    
    if Path(test_path).parent.joinpath('transcripts.txt').exists():
        lbl_path = Path(test_path).parent.joinpath('transcripts.txt')
        transcripts = [line.strip() for line in open(lbl_path, 'r', encoding='utf-8').readlines()]
        info = {}
        for line in transcripts:
            nameNlbl = line.split('\t')
            if len(nameNlbl) > 1:
                info[nameNlbl[0]] = nameNlbl[1]
            
        check_lbl = True
    
    my_model = CustomEncoderDecoderASR.from_hparams(
        source=model_path, 
        hparams_file=config_path, 
        savedir="pretrained_model", 
        run_opts={'device':args.device}    
    )
            
    if check_lbl:
        save_df = {'id':[], 'transcript': [], 'iscorrect': [], 'wer': []}
    else:
        save_df = {'id':[], 'transcript': []}
        
    if get_conf:
        save_df.update({'conf': []})
    
    count = 0
    if batch_size == 1:
        for audio_path in test_list:
            #audio is already normalized with Pretrained class 
            # self.audio_normalizer = hparams.get(
            #    "audio_normalizer", AudioNormalizer())
            pred_transcript = my_model.transcribe_file(audio_path.as_posix())
            print(f'Audio {count+1} ', audio_path.name, ':', pred_transcript)
            count+=1
            total_correct = update_df(audio_path.stem, save_df, check_lbl, total_correct, pred_transcript)
    else:
        batch_id = build_indices_plain(test_list, batch_size)
        if start_idx == -1:
            batch_id = batch_id[::-1]
            start_idx=0
            
        for idx, batch in enumerate(batch_id[start_idx:end_idx]):
            idx += start_idx
            
            paths = [test_list[id_] for id_ in batch]
            log_probs = [None]*len(paths)
            if get_conf:
                pred_transcripts, log_probs = my_model.transcribe_batch_file(paths=paths, get_conf=get_conf)
                log_probs = [torch.tensor(log_prob[0]) for log_prob in log_probs]
                log_probs = [torch.exp(log_prob).cpu() for log_prob in log_probs]
            else:
                pred_transcripts = my_model.transcribe_batch_file(paths=paths, get_conf=get_conf)
                
            for pred, path, log_prob in zip(pred_transcripts, paths, log_probs):
                print(f'Audio {count+1} ', path.name, ':', str(pred))
                count+=1
                total_correct = update_df(path.stem, save_df, check_lbl, total_correct, str(pred), log_prob)
                
    save_df=pd.DataFrame(save_df)
    save_df.to_csv(Path(submission_path).joinpath('results.csv'), index=False, header=True, sep='\t')
    if check_lbl:
        print('ACC: ', float(total_correct*100.0/len(test_list)))
        print('avg WER', sum(save_df['wer'])/len(test_list))
