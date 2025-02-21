# python run.py --input_dir /workspace/AcademiCodec/egs/HiFi-Codec-24k-240d/outputhf-dummy --output_dir /workspace/wavmark/outputhf-dummy --csv /workspace/wavmark/ber-outputhf-dummy.csv > 20250218.log
import argparse
import os
import torch
import numpy as np
import soundfile as sf
import wavmark
import pandas as pd

def process_audio_folder(input_folder, output_folder, csv_file):
    """Encodes and decodes messages for every 1-second segment of each `.wav` file and saves BER & concatenated messages to CSV."""

    # Load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = wavmark.load_model().to(device)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get list of `.wav` files in the folder
    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]

    if not audio_files:
        print(f"No audio files found in '{input_folder}'.")
        return

    results = []

    for audio_file in audio_files:
        input_path = os.path.join(input_folder, audio_file)
        output_path = os.path.join(output_folder, audio_file)

        print(f"Processing: {input_path}")

        # Load audio file
        signal, sample_rate = sf.read(input_path)

        # Ensure the sample rate is 16kHz (if different, warn but continue)
        if sample_rate != 16000:
            print(f"Warning: {audio_file} has a sample rate of {sample_rate} Hz, expected 16,000 Hz.")

        segment_length = 16000  # 1 second = 16,000 samples
        num_segments = len(signal) // segment_length  # Number of full 1s segments
        remainder = len(signal) % segment_length  # Remaining samples

        encoded_signal = []  # Store full processed audio
        total_ber = 0  # Accumulate BER across segments
        segment_results = []  # Store per-segment results
        concatenated_messages = ""  # Store all messages for the entire audio file

        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length
            segment = signal[start:end]

            # Generate random 32-bit message for each segment
            message_npy = np.random.choice([0, 1], size=32)

            # Encode
            with torch.no_grad():
                signal_tensor = torch.FloatTensor(segment).to(device)[None]
                message_tensor = torch.FloatTensor(message_npy).to(device)[None]
                signal_wmd_tensor = model.encode(signal_tensor, message_tensor)
                signal_wmd_npy = signal_wmd_tensor.detach().cpu().numpy().squeeze()

            encoded_signal.append(signal_wmd_npy)  # Store encoded segment

            # Decode
            with torch.no_grad():
                signal_tensor = torch.FloatTensor(signal_wmd_npy).to(device).unsqueeze(0)
                message_decoded_npy = (model.decode(signal_tensor) >= 0.5).int().detach().cpu().numpy().squeeze()

            # Calculate Bit Error Rate (BER)
            BER = (message_npy != message_decoded_npy).mean() * 100
            total_ber += BER

            # Convert message to string
            message_str = "".join(map(str, message_npy.tolist()))
            concatenated_messages += message_str  # Append message for full audio

            # Save per-segment result
            segment_results.append({
                "filename": audio_file,
                "segment": i + 1,
                "BER": BER,
                "message": message_str  # Save message as a string of 0s and 1s
            })

            print(f"Segment {i+1}/{num_segments} - BER: {BER:.2f}%")

        # Handle remainder samples (if audio length is not an exact multiple of 1 second)
        if remainder > 0:
            print(f"Processing remainder segment ({remainder} samples)")
            remainder_signal = signal[-remainder:]

            # Pad to 1 second if necessary
            padded_signal = np.pad(remainder_signal, (0, segment_length - remainder), mode='constant')

            # Generate random message for remainder
            message_npy = np.random.choice([0, 1], size=32)

            # Encode
            with torch.no_grad():
                signal_tensor = torch.FloatTensor(padded_signal).to(device)[None]
                message_tensor = torch.FloatTensor(message_npy).to(device)[None]
                signal_wmd_tensor = model.encode(signal_tensor, message_tensor)
                signal_wmd_npy = signal_wmd_tensor.detach().cpu().numpy().squeeze()

            # Trim back to original remainder length
            encoded_signal.append(signal_wmd_npy[:remainder])

            # Decode
            with torch.no_grad():
                signal_tensor = torch.FloatTensor(signal_wmd_npy).to(device).unsqueeze(0)
                message_decoded_npy = (model.decode(signal_tensor) >= 0.5).int().detach().cpu().numpy().squeeze()

            # Calculate BER
            BER = (message_npy != message_decoded_npy).mean() * 100
            total_ber += BER
            num_segments += 1  # Count the remainder segment

            message_str = "".join(map(str, message_npy.tolist()))
            concatenated_messages += message_str  # Append to full message

            segment_results.append({
                "filename": audio_file,
                "segment": num_segments,
                "BER": BER,
                "message": message_str
            })

            print(f"Remainder segment - BER: {BER:.2f}%")

        # Concatenate all encoded segments into a full audio
        final_encoded_audio = np.concatenate(encoded_signal)

        # Save encoded audio
        sf.write(output_path, final_encoded_audio, sample_rate)
        print(f"Saved encoded audio: {output_path}")

        # Store final average BER and concatenated message
        avg_ber = total_ber / num_segments  # Average BER across all segments
        segment_results.append({
            "filename": audio_file,
            "segment": "ALL",
            "BER": avg_ber,
            "message": concatenated_messages  # Full concatenated message
        })
        print(f"✅ Final Average BER for {audio_file}: {avg_ber:.2f}%\n")

        results.extend(segment_results)  # Append all segment results

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    print(f"📄 BER results and messages saved to: {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply watermarking encoding and decoding to all audio files in a folder, processing per 1-second segment.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the folder containing input audio files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the folder where encoded files will be saved.")
    parser.add_argument("--csv", type=str, required=True, help="Path to save BER results in a CSV file.")

    args = parser.parse_args()

    process_audio_folder(args.input_dir, args.output_dir, args.csv)