import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import kenlm
import joblib
import yaml
import numpy as np
import json
import argparse


def model_selection(results, test_path):
    "Combine result from multiple model baseed on prediction confidence"
    
    audio_list = Path(test_path).rglob('*.wav')
    final_result = defaultdict(list)
    
    all_ids = []
    all_dfs = []
    for idx, path_ in enumerate(results):
        df = pd.read_csv(path_, sep='\t')
        all_dfs.append(df)
        ids = df['id'].values.tolist()
        ids = sorted(ids)
        all_ids.append((ids, idx))
    
    for audio in audio_list:
        audio_name = audio.stem
        candidate_audios = []
        for ids in all_ids:
            if audio_name in ids[0]:
                candidate_audios.append(ids[1])     

        confs = []
        for can in candidate_audios:
            df = all_dfs[can]
            row = df.loc[df['id']==audio_name]
            conf = float(row['conf'])
            confs.append((can, conf))
            
        choose_model = max(confs, key=lambda x: x[1])[0]

        pred = all_dfs[choose_model].loc[all_dfs[choose_model]['id']==audio_name, 'transcript']
        pred = pred.values.tolist()[0]
        final_result['id'].append(audio_name)
        final_result['pred'].append(str(pred))
    
    final_result = pd.DataFrame(final_result)
    
    return final_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args=parser.parse_args()
        
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    final_result = model_selection(
        config['model_info'], 
        config['test_path']
    )

    output_path = Path(config['output_path'])
    output_path.mkdir(parents=True)

    final_result.to_csv(output_path.joinpath('results.csv').as_posix(), sep='\t', index=False, header=False)

    with open(output_path.joinpath('config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
