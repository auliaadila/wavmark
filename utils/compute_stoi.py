# Compute STOI
import argparse
import glob
import os

import numpy as np
from pystoi import stoi
from scipy.io import wavfile
from tqdm import tqdm
import pandas as pd


def cal_stoi_audio(ref_signal, deg_signal, sample_rate):
    min_len = min(len(ref_signal), len(deg_signal))
    ref = ref_signal[:min_len]
    deg = deg_signal[:min_len]
    cur_stoi = stoi(ref, deg, sample_rate, extended=False)

    return cur_stoi

def cal_stoi(ref_dir, deg_dir, csv):
    input_files = glob.glob(f"{deg_dir}/*.wav")

    if len(input_files) < 1:
        raise RuntimeError(f"Found no wavs in {ref_dir}")

    results = []
    stoi_scores = []

    for deg_wav in tqdm(input_files):
        ref_wav = os.path.join(ref_dir, os.path.basename(deg_wav))
        rate, ref = wavfile.read(ref_wav)
        rate, deg = wavfile.read(deg_wav)
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]
        cur_stoi = stoi(ref, deg, rate, extended=False)
        
        if(csv):
             # Prepare result dictionary
            result = {
                "filename": ref_wav,
                "stoi": round(cur_stoi, 3)
            }
            results.append(result)

        stoi_scores.append(cur_stoi)
    
    if(csv):
        # Ensure the directory exists before saving the CSV
        csv_dir = os.path.dirname(csv)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)
            print(f"Created directory: {csv_dir}")

        # Save results to CSV
        df = pd.DataFrame(results)
        df.to_csv(csv, index=False)
        print(f"STOI results saved to: {csv}")

    return np.mean(stoi_scores)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute STOI measure")

    parser.add_argument(
        '-r', '--ref_dir', required=True, help="Reference wave folder.")
    parser.add_argument(
        '-d', '--deg_dir', required=True, help="Degraded wave folder.")
    parser.add_argument('-s', '--csv', required=True, help="Saved CSV file path")

    args = parser.parse_args()

    stoi_score = cal_stoi(args.ref_dir, args.deg_dir, args.csv)
    print(f"STOI: {stoi_score}")

## Results
# hifi-clean-2
# python compute_stoi.py -r /workspace/AcademiCodec/dataset/LibriTTS/test-clean-2 -d /workspace/AcademiCodec/egs/HiFi-Codec-24k-240d/output/hf -s /workspace/AcademiCodec/egs/HiFi-Codec-24k-240d/results/stoi-hifi-clean-2.csv
# STOI: 0.9380094691230352
# NB PESQ: 3.9291073006552617
# WB PESQ: 3.73306757372778

# hifi-wavmark-clean-2
# python compute_stoi.py -r /workspace/AcademiCodec/dataset/LibriTTS/test-clean-2 -d /workspace/wavmark/output/hifi-wavmark-clean-2 -s /workspace/wavmark/results/stoi-hifi-wavmark-clean-2.csv
# STOI: 0.9429923506524662
# NB PESQ: 1.683132453279181
# WB PESQ: 1.506200950879317

# wavmark-clean-2
# python compute_stoi.py -r /workspace/AcademiCodec/dataset/LibriTTS/test-clean-2 -d /workspace/wavmark/output/wavmark-clean-2 -s /workspace/wavmark/results/stoi-wavmark-clean-2.csv
# STOI: 0.9992303874943328
# NB PESQ: 1.761982119127071
# WB PESQ: 1.6130516998497122