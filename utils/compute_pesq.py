# Compute PESQ (WB: wide band, NB: narrow band)
import argparse
import glob
import os

import scipy.signal as signal
from pesq import pesq
from scipy.io import wavfile
from tqdm import tqdm
import pandas as pd

def cal_pesq_audio(ref_signal, ref_rate, deg_signal, deg_rate):
    if ref_rate != 16000:
        ref = signal.resample(ref_signal, 16000)
    if deg_rate != 16000:
        deg = signal.resample(deg_signal, 16000)
    
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]

    nb = pesq(16000, ref, deg, 'nb')
    wb = pesq(16000, ref, deg, 'wb')

    return nb, wb

def cal_pesq(ref_dir, deg_dir, csv):
    input_files = glob.glob(f"{deg_dir}/*.wav")

    nb_pesq_scores = 0.0
    wb_pesq_scores = 0.0

    results = []

    for deg_wav in tqdm(input_files):
        ref_wav = os.path.join(ref_dir, os.path.basename(deg_wav))
        ref_rate, ref = wavfile.read(ref_wav)
        deg_rate, deg = wavfile.read(deg_wav)
        if ref_rate != 16000:
            ref = signal.resample(ref, 16000)
        if deg_rate != 16000:
            deg = signal.resample(deg, 16000)

        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]

        nb_pesq = pesq(16000, ref, deg, 'nb')
        wb_pesq = pesq(16000, ref, deg, 'wb')

        if(csv):
             # Prepare result dictionary
            result = {
                "filename": ref_wav,
                "pesq_nb": round(nb_pesq, 3),
                "pesq_wb": round(wb_pesq, 3)
            }
            results.append(result)

        nb_pesq_scores += nb_pesq
        wb_pesq_scores += wb_pesq

    if(csv):
        # Ensure the directory exists before saving the CSV
        csv_dir = os.path.dirname(csv)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)
            print(f"Created directory: {csv_dir}")

        # Save results to CSV
        df = pd.DataFrame(results)
        df.to_csv(csv, index=False)
        print(f"PESQ results saved to: {csv}")


    return nb_pesq_scores / len(input_files), wb_pesq_scores / len(input_files)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute PESQ measure.")

    parser.add_argument(
        '-r', '--ref_dir', required=True, help="Reference wave folder.")
    parser.add_argument(
        '-d', '--deg_dir', required=True, help="Degraded wave folder.")
    parser.add_argument('-s', '--csv', required=True, help="Saved CSV file path")

    args = parser.parse_args()

    nb_score, wb_score = cal_pesq(args.ref_dir, args.deg_dir, args.csv)
    print(f"NB PESQ: {nb_score}")
    print(f"WB PESQ: {wb_score}")

## Results
# hifi-clean-2
# python compute_pesq.py -r /workspace/AcademiCodec/dataset/LibriTTS/test-clean-2 -d /workspace/AcademiCodec/egs/HiFi-Codec-24k-240d/output/hf -s /workspace/AcademiCodec/egs/HiFi-Codec-24k-240d/results/pesq-hifi-clean-2.csv
# NB PESQ: 3.9291073006552617
# WB PESQ: 3.73306757372778

# hifi-wavmark-clean-2
# python compute_pesq.py -r /workspace/AcademiCodec/dataset/LibriTTS/test-clean-2 -d /workspace/wavmark/output/hifi-wavmark-clean-2 -s /workspace/wavmark/results/pesq-hifi-wavmark-clean-2.csv
# NB PESQ: 1.683132453279181
# WB PESQ: 1.506200950879317

# wavmark-clean-2
# python compute_pesq.py -r /workspace/AcademiCodec/dataset/LibriTTS/test-clean-2 -d /workspace/wavmark/output/wavmark-clean-2 -s /workspace/wavmark/results/pesq-wavmark-clean-2.csv
# NB PESQ: 1.761982119127071
# WB PESQ: 1.6130516998497122