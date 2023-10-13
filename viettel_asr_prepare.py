import os
import json
import logging
from speechbrain.dataio.preprocess import AudioNormalizer
import random
from pathlib import Path
import numpy as np
import torchaudio
import re
import json
from tqdm import tqdm
import argparse
import pandas as pd
import shutil
import kenlm
import pandas as pd

random.seed(1999)
logger = logging.getLogger(__name__)
TRAIN_RATIO = 0.85
SAMPLERATE = 16000
MAX_DURATION = 30 #s


chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\']'

def remove_special_characters(transcript):
    transcript = re.sub(chars_to_ignore_regex, '', transcript).lower()
    return transcript


def split_train_val(transcript_name, data_folder, noise_list):
    lines = [line for line in open(Path(data_folder).parent.joinpath(transcript_name), 'r', encoding='utf-8').readlines() if len(line.strip().split('\t'))>1]
    transcripts = {}
    
    for line in lines:
        nameNtranscript = line.strip().split('\t')
        if noise_list and nameNtranscript[0] in noise_list:
            continue
        transcripts[nameNtranscript[0]] = nameNtranscript[1]

    all_durations = {}
    
    for i, audio in enumerate(transcripts):
        audio_path = Path(data_folder).joinpath(audio).with_suffix('.wav')
        wav_meta = torchaudio.info(audio_path.as_posix())
        all_durations[audio] = wav_meta.num_frames / wav_meta.sample_rate
    
    new_transcript = []
    for audio_name in all_durations:
        if all_durations[audio_name] <= MAX_DURATION:
            transcript = audio_name + '\t' + transcripts[audio_name]
            new_transcript.append(transcript)

    random.shuffle(new_transcript)
            
    train_transcripts = new_transcript[:int(TRAIN_RATIO*len(new_transcript))]
    val_transcripts = new_transcript[int(TRAIN_RATIO*len(new_transcript)):]
    return train_transcripts, val_transcripts


def create_pseudo_labels(pseudo_data_path, real_data_path, pseudo_label_csv, train_json_path, valid_json_path, save_labeled_path=None):
    "Combine pseudo label of training100h with current training data from training30h"
    pseudo_label_df = pd.read_csv(pseudo_label_csv, sep='\t', names=['id', 'pred'])
    
    with open(train_json_path, 'r', encoding='utf-8') as f:
        training_labeled = json.load(f)
    
    with open(valid_json_path, 'r', encoding='utf-8') as f:
        valid_labeled = json.load(f)
    
    pseudo_label_dict = {}
    for idx, row in pseudo_label_df.iterrows():
        wav_file = Path(pseudo_data_path).joinpath(row['id']).with_suffix('*.wav').as_posix()
        signal, sr = torchaudio.load(wav_file)
        signal = AudioNormalizer(SAMPLERATE)(signal, sr)
        assert signal.dim() == 1
        duration = signal.shape[0] / SAMPLERATE

        # Manipulate path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext('/'.join(path_parts[-4:]))
        relative_path = os.path.join("{data_root}", '/'.join(path_parts[-4:]))
        
        pseudo_label_dict[uttid] = {
            "wav": relative_path,
            "length": duration,
            "words": str(row['pred']),
        }

    new_training_data = {}
    for item in training_labeled:
        new_item = Path(real_data_path).joinpath(item).with_suffix('*.wav').as_posix()
        new_path = os.path.join("{data_root}",real_data_path,training_labeled[item]['wav'].split('/')[-1])
        new_training_data[new_item] = {
            'wav': new_path,
            'length': training_labeled[item]['length'],
            'words': training_labeled[item]['words']
        }
        
    new_valid_data = {}
    for item in valid_labeled:
        new_item = Path(real_data_path).joinpath(item).with_suffix('*.wav').as_posix()
        new_path = os.path.join("{data_root}",real_data_path,valid_labeled[item]['wav'].split('/')[-1])
        new_valid_data[new_item] = {
            'wav': new_path,
            'length': valid_labeled[item]['length'],
            'words': valid_labeled[item]['words']
        }
    
    final_train_dict = {**pseudo_label_dict, **new_training_data}
    keys = list(final_train_dict.keys())
    random.shuffle(keys)
    final_train_dict = dict([(k,final_train_dict[k]) for k in keys])
    
    # Writing the dictionary to the json file
    with open(Path(save_labeled_path).joinpath('train.json').as_posix(), mode="w", encoding='utf-8') as json_f:
        json.dump(final_train_dict, json_f, indent=2, ensure_ascii=False)
    
    # Writing the dictionary to the json file
    with open(Path(save_labeled_path).joinpath('valid.json').as_posix(), mode="w", encoding='utf-8') as json_f:
        json.dump(new_valid_data, json_f, indent=2, ensure_ascii=False)
        
    
def prepare_viettel_asr(
    transcript_name, data_folder, save_json_train, save_json_valid, multi_folder=False, noise_list=None
):
    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}"
    )
    extensions = ".wav"
    
    train_transcript, val_transcript = split_train_val(transcript_name, data_folder, noise_list)
    train_trans_dict = get_transcription(train_transcript)
    val_trans_dict = get_transcription(val_transcript)
    
    wav_list_train = []
    for sample in train_trans_dict:
        if len(sample) > 0:
            wav_file = Path(data_folder) / sample
            wav_file = wav_file.with_suffix(extensions).as_posix()
            wav_list_train.append(wav_file)
            
    wav_list_val = []
    for sample in val_trans_dict:
        if len(sample) > 0:
            wav_file = Path(data_folder) / sample
            wav_file = wav_file.with_suffix(extensions).as_posix()
            wav_list_val.append(wav_file)
            
    # Create the json files
    create_json(multi_folder, wav_list_train, train_trans_dict, save_json_train)
    create_json(multi_folder, wav_list_val, val_trans_dict, save_json_valid)
    
    
def get_transcription(trans_list):
    """
    Returns a dictionary with the transcription of each sentence in the dataset.

    Arguments
    ---------
    trans_list : list of str
        The list of transcription files.
    """
    # Processing all the transcription files in the list
    trans_dict = {}
    for trans_item in trans_list:
        # Reading the text file
        trans_info=trans_item.strip().split("\t")
        if len(trans_info) > 0:
            uttid = trans_info[0]
            text = trans_info[-1]
            trans_dict[uttid] = text

    logger.info("Transcription files read!")
    return trans_dict


def create_json(multi_folder, wav_list, trans_dict, json_file):
    """
    Creates the json file given a list of wav files and their transcriptions.

    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    trans_dict : dict
        Dictionary of sentence ids and word transcriptions.
    json_file : str
        The path of the output json file
    """
    # Processing all the wav files in the list
    json_dict = {}
    for wav_file in wav_list:
        # Reading the signal (to retrieve duration in seconds)
        signal, sr = torchaudio.load(wav_file)
        signal = AudioNormalizer(SAMPLERATE)(signal, sr)
        assert signal.dim() == 1
        duration = signal.shape[0] / SAMPLERATE

        # Manipulate path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        if multi_folder:            
            uttid, _ = os.path.splitext('/'.join(path_parts[-4:]))
            relative_path = os.path.join("{data_root}", '/'.join(path_parts[-4:]))
        else:
            uttid, _ = os.path.splitext(path_parts[-1])
            relative_path = os.path.join("{data_root}", path_parts[-1])
            
        preprocess_text = remove_special_characters(trans_dict[uttid])
        
        # Create entry for this utterance
        json_dict[uttid] = {
            "wav": relative_path,
            "length": duration,
            "words": preprocess_text,
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w", encoding='utf-8') as json_f:
        json.dump(json_dict, json_f, indent=2, ensure_ascii=False)

    logger.info(f"{json_file} successfully created!")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def get_pseudo_by_conf(pseudo_csv_path, training100h_path, conf=0.75):
    df = pd.read_csv(pseudo_csv_path, sep='\t')

    conf75_df = df[df['conf']>0.75]
    conf75_df = conf75_df[['id', 'transcript']]
    conf75_df_id = conf75_df['id'].values.tolist()


    training_100h = [line.strip().split('\t') for line in open(training100h_path, 'r', encoding='utf-8').readlines() if len(line.strip().split('\t'))>1]
    training_100h = list(zip(*training_100h))
    training_100h = pd.DataFrame({'id':training_100h[0], 'transcript':training_100h[-1]})
    sub_100h = training_100h[training_100h['id'].isin(conf75_df_id)]

    sub_100h.reset_index(drop=True, inplace=True)
    conf75_df.reset_index(drop=True, inplace=True)

    assert len(conf75_df) == len(sub_100h)

    combine = sub_100h.merge(conf75_df, how='inner', on='id', suffixes=['_gt', '_pred'])
    combine = combine.reset_index(drop=True)

    return combine


def kenlm_postprocess(combine_df, kenlm_model, save_path):
    samples = combine_df[~combine_df.isna().any(axis=1)]

    for idx, row in samples.iterrows():
        pred = row['transcript_pred']
        pred_score = kenlm_model.score(pred)
        ori_pred_score = kenlm_model.score(row['transcript_gt'])
        if np.abs(pred_score) < np.abs(ori_pred_score):
            samples.loc[idx, 'choose_pred'] = row['transcript_pred']
        else:
            samples.loc[idx, 'choose_pred'] = row['transcript_gt']
        
    postprocess_kenlm = samples[['id', 'choose_pred']]
    postprocess_kenlm.to_csv(Path(save_path).joinpath('kenlm_postprocess.csv').as_posix(), index=False, header=False, sep='\t')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build_option', type=int, help='Build real data from training 30h or build combine with pseudo label')
    parser.add_argument('--transcript_name', default='transcripts.txt', help='Name of the transcript file in the dataset (30h, 100h)')
    parser.add_argument('--data_folder', default='dataset/raw_data/dataset1/training_30h/wavs/', help='Path to wav audio file of dataset')
    parser.add_argument('--save_json_train', default='dataset/processed_label/training30h/train.json')
    parser.add_argument('--save_json_valid', default='dataset/processed_label/training30h/valid.json')
    
    
    parser.add_argument('--pseudo_data_path', default='dataset2/training_100h/wavs')
    parser.add_argument('--real_data_path', default='dataset/raw_data/dataset1/training_30h/wavs/')
    parser.add_argument('--pseudo_label_csv', default='dataset/pseudo_label/v1/kenlm_postprocess.csv')
    parser.add_argument('--save_labeled_path', default='dataset/processed_label/combine_label/v1')
    
    parser.add_argument('--kenlm_weight', help='Path to weight of kenlm language model')
    parser.add_argument('--pseudo_label_raw', help='Path to pseudo label get by model inference', default='dataset/pseudo_label/v1/results.csv')
    
    args = parser.parse_args()
    
    if args.build_option == 1:
        "Create label for training 30h dataset"
        prepare_viettel_asr(
            args.transcript_name,
            args.data_folder,
            args.save_json_train,
            args.save_json_valid
        )
    elif args.build_option == 2:
        "Create label which combine pseudo label from model after kenlm postprocess and previous training 30h dataset"
        create_pseudo_labels(
            args.pseudo_data_path,
            args.real_data_path,
            args.pseudo_label_csv,
            args.save_json_train,
            args.save_json_valid,
            args.save_labeled_path
        )
    elif args.build_option == 3:
        "Load pseudo label file and filter by confidence and postprocess wiht kenlm"
        model = kenlm.LanguageModel(args.kenlm_weight)
        combine_df = get_pseudo_by_conf(args.pseudo_label_raw, args.pseudo_data_path, conf=0.75)
        kenlm_postprocess(combine_df, model, Path(args.pseudo_label_raw).parent.as_posix())
    else:
        raise ValueError('Not exist option')