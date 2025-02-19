# Audio -> watermark embedding (using wavmark) -> attack -> watermark detection -> BER
import argparse
import os
import torch
import numpy as np
import soundfile as sf
import wavmark
import pandas as pd
import logging
import datetime
from utils.signal_attacks import SignalProcessingAttacks as spa
from utils.compute_pesq import cal_pesq_audio
from utils.compute_stoi import cal_stoi_audio


def setup_logging(output_dir):
    """Sets up logging with a timestamped log file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S")
    log_file = os.path.join(output_dir, f"run_attack_{timestamp}.log")
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Prints logs to console as well
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")


def run_attack(a):
    """Runs watermark embedding, attack application, decoding, and BER calculation."""
    
    setup_logging(a.output_dir)
    logging.info("Starting watermarking and attack process...")

    # Load Wavmark model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = wavmark.load_model().to(device)
    logging.info(f"Wavmark model loaded on {device}")

    # Ensure output folder exists
    os.makedirs(a.output_dir, exist_ok=True)

    # Get list of `.wav` files in the folder
    audio_files = [f for f in os.listdir(a.input_dir) if f.endswith('.wav')]

    if not audio_files:
        logging.warning(f"No audio files found in '{a.input_dir}'.")
        return
    
    results = []

    for audio_file in audio_files:
        input_path = os.path.join(a.input_dir, audio_file)
        logging.info(f"Processing: {input_path}")

        # Load and process entire audio
        signal, sample_rate = sf.read(input_path)
        num_chunks = len(signal) // 16000

        if num_chunks == 0:
            logging.warning(f"{audio_file} is too short for watermarking, skipping.")
            continue

        # Initialize message storage
        embedded_messages = []
        decoded_messages = []
        watermarked_audio = []

        for i in range(num_chunks):
            chunk = signal[i * 16000 : (i + 1) * 16000]
            message_npy = np.random.choice([0, 1], size=32)
            embedded_messages.extend(message_npy)

            # Encode watermark
            with torch.no_grad():
                signal_tensor = torch.FloatTensor(chunk).to(device)[None]
                message_tensor = torch.FloatTensor(message_npy).to(device)[None]
                signal_wmd_tensor = model.encode(signal_tensor, message_tensor)
                signal_wmd_npy = signal_wmd_tensor.detach().cpu().numpy().squeeze()
                watermarked_audio.append(signal_wmd_npy)   

            # Decode watermark
            with torch.no_grad():
                signal_tensor = torch.FloatTensor(signal_wmd_npy).to(device).unsqueeze(0)
                message_decoded_npy = (model.decode(signal_tensor) >= 0.5).int().detach().cpu().numpy().squeeze()
                decoded_messages.extend(message_decoded_npy)
        
        full_watermarked_audio = np.concatenate(watermarked_audio)

        # Save watermarked audio if enabled
        if a.save_watermarked:
            watermarked_path = os.path.join(a.output_dir, f"wm_{audio_file}")
            sf.write(watermarked_path, full_watermarked_audio, sample_rate)
            logging.info(f"Saved watermarked audio: {watermarked_path}")
        
        # Compute PESQ
        try:
            nb_score, wb_score = cal_pesq_audio(signal, sample_rate, full_watermarked_audio, sample_rate)
            # nb_pesq = pesq(16000, ref_signal, deg_signal, 'nb')
            # wb_pesq = pesq(16000, ref_signal, deg_signal, 'wb')
        except Exception as e:            
            logging.error(f"Error computing PESQ for signals: {e}")
            nb_score, wb_score = np.nan, np.nan
        
        # nb_score, wb_score = cal_pesq_audio(signal, sample_rate, full_watermarked_audio, sample_rate)

        # Compute STOI
        try:
            stoi_score = cal_stoi_audio(signal, full_watermarked_audio, sample_rate)
        except Exception as e:            
            logging.error(f"Error computing STOI for signals: {e}")
            stoi_score = np.nan

        # Compute BER before attacks
        embedded_messages = np.array(embedded_messages)
        decoded_messages = np.array(decoded_messages)
        BER = (embedded_messages != decoded_messages).mean() * 100

        # Prepare result dictionary
        attack_results = {
            "filename": audio_file,
            "pesq_nb": round(nb_score, 3),
            "pesq_wb": round(wb_score, 3),
            "stoi": round(stoi_score, 3),
            "ber": round(BER, 3),
            "gaussian_noise": np.nan, "reverb": np.nan, "mp3": np.nan, "ogg": np.nan,
            "mp4": np.nan, "resample": np.nan, "requant_8bit": np.nan, "requant_16bit": np.nan,
            "pitch_shift": np.nan, "low_pass": np.nan, "g711": np.nan, "g726": np.nan
        }
        
        # Apply attack
        if a.is_attack:
            wm_audio = np.concatenate(watermarked_audio)
            attacks = spa(wm_audio, sample_rate)
            attack_methods = {
                "gaussian_noise": attacks.add_gaussian_noise(),
                "reverb": attacks.apply_reverb(),
                "mp3": attacks.mp3_compression(),
                "ogg": attacks.ogg_compression(),
                "mp4": attacks.mp4_compression(),
                "resample": attacks.apply_resampling(),
                "requant_8bit": attacks.requantization(8),
                "requant_16bit": attacks.requantization(16),
                "pitch_shift": attacks.pitch_shift(),
                "low_pass": attacks.low_pass_filter(),
                "g711": attacks.g711_codec_attack(),
                "g726": attacks.g726_codec_attack()
            }
            for attack_name, attacked_signal in attack_methods.items():
                if attacked_signal is None or len(attacked_signal) == 0:
                    logging.warning(f"{audio_file} - Attack {attack_name} failed. Setting BER to NaN.")
                    attack_results[attack_name] = np.nan
                    continue

                if a.save_attacked:
                    attack_output_dir = os.path.join(a.output_dir, attack_name)
                    attacks.save_audio(attack_output_dir, audio_file, attacked_signal)
                
                if len(attacked_signal) % 16000 != 0:
                    logging.warning(f"{audio_file} - Attack {attack_name} caused signal length mismatch: {len(wm_audio)} -> {len(attacked_signal)}. Setting BER to NaN.")
                    attack_results[attack_name] = np.nan
                    continue
                
                # Decode watermark from attacked audio (per chunk)
                attack_decoded_messages = []
                
                for i in range(num_chunks):
                    chunk = attacked_signal[i * 16000 : (i + 1) * 16000]

                    with torch.no_grad():
                        signal_tensor = torch.FloatTensor(chunk).to(device).unsqueeze(0)
                        message_decoded_npy = (model.decode(signal_tensor) >= 0.5).int().detach().cpu().numpy().squeeze()

                    attack_decoded_messages.append(message_decoded_npy)

                attack_messages = np.concatenate(attack_decoded_messages)  
                attack_messages = np.array(attack_messages)
                BER_attack = (embedded_messages != attack_messages).mean() * 100

                logging.info(f"{audio_file} - Attack: {attack_name} | BER: {BER_attack:.2f}%")
                
                attack_results[attack_name] = round(BER_attack, 3)

            results.append(attack_results)
        else: 
            results.append(attack_results)


    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(a.csv, index=False)
    logging.info(f"BER results saved to: {a.csv}")


def main():
    parser = argparse.ArgumentParser(description="Watermark, attack, and decode audio files.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--is_attack", action="store_true")
    parser.add_argument("--save_watermarked", action="store_true")
    parser.add_argument("--save_attacked", action="store_true")
    
    a = parser.parse_args()
    run_attack(a)


if __name__ == "__main__":
    main()

## Usage
# [dummy] python runattack.py --input_dir /workspace/AcademiCodec/egs/HiFi-Codec-24k-240d/outputhf-dummy --output_dir /workspace/wavmark/output/outputhf-dummy --csv /workspace/wavmark/results/ber-attack-outputhf-dummy.csv --is_attack --save_watermarked --save_attacked
# [test-2; hf pretrained] python runattack.py --input_dir /workspace/AcademiCodec/egs/HiFi-Codec-24k-240d/outputhf-2 --output_dir /workspace/wavmark/output/outputhf-2 --csv /workspace/wavmark/results/ber-attack-outputhf-2.csv --is_attack --save_watermarked --save_attacked
    
# python runattack.py --input_dir /workspace/AcademiCodec/egs/HiFi-Codec-24k-240d/output-2 --output_dir /workspace/wavmark/output/output-2 --csv /workspace/wavmark/results/ber-attack-output-2.csv --is_attack --save_watermarked --save_attacked