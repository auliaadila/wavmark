import argparse
import numpy as np
import librosa
import librosa.display
import scipy.signal
import soundfile as sf
from pydub import AudioSegment
import tempfile
import os
import subprocess

class SignalProcessingAttacks:
    def __init__(self, waveform, sample_rate):
        self.y = waveform
        self.sr = sample_rate

    def add_gaussian_noise(self, snr=20):
        rms_signal = np.sqrt(np.mean(self.y**2))
        rms_noise = rms_signal / (10**(snr / 20))
        noise = np.random.normal(0, rms_noise, self.y.shape)
        return np.clip(self.y + noise, -1, 1)

    def apply_reverb(self, room_size=0.6):
        b = np.array([1] + [0] * int(self.sr * room_size) + [-0.5])
        return scipy.signal.lfilter(b, [1], self.y)

    def mp3_compression(self, bitrate="32k"):
        return self._apply_compression("mp3", bitrate)

    def ogg_compression(self, bitrate="32k"):
        return self._apply_compression("ogg", bitrate)

    def mp4_compression(self, bitrate="32k"):
        return self._apply_compression("mp4", bitrate)

    def _apply_compression(self, format, bitrate):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            sf.write(tmp_wav.name, self.y, self.sr)

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}") as tmp_compressed:
            audio = AudioSegment.from_wav(tmp_wav.name)
            audio.export(tmp_compressed.name, format=format, bitrate=bitrate)
            compressed_audio = AudioSegment.from_file(tmp_compressed.name, format=format)
            y_compressed = np.array(compressed_audio.get_array_of_samples()).astype(np.float32) / (2**15)

        os.remove(tmp_wav.name)
        os.remove(tmp_compressed.name)
        return y_compressed

    def apply_resampling(self, new_sr=8000):
        downsampled = librosa.resample(self.y, orig_sr=self.sr, target_sr=new_sr)
        return librosa.resample(downsampled, orig_sr=new_sr, target_sr=self.sr)

    def requantization(self, bit_depth=8):
        max_val = 2**(bit_depth - 1)
        y_requantized = np.round(self.y * max_val) / max_val
        return np.clip(y_requantized, -1, 1)

    def time_stretch(self, rate=1.2):
        return librosa.effects.time_stretch(self.y, rate)

    def pitch_shift(self, n_steps=2):
        return librosa.effects.pitch_shift(self.y, sr=self.sr, n_steps=n_steps)

    def low_pass_filter(self, cutoff_freq=3000):
        nyquist = 0.5 * self.sr
        norm_cutoff = cutoff_freq / nyquist
        b, a = scipy.signal.butter(4, norm_cutoff, btype='low')
        return scipy.signal.lfilter(b, a, self.y)

    def g711_codec_attack(self):
        return self._apply_ffmpeg_codec("pcm_mulaw", "g711_output.wav")

    def g726_codec_attack(self, bitrate="32k"):
        return self._apply_ffmpeg_codec("g726", "g726_output.wav", bitrate)

    def _apply_ffmpeg_codec(self, codec, output_filename, bitrate=None):
        temp_input = "temp_input.wav"
        temp_output = output_filename

        sf.write(temp_input, self.y, self.sr)

        command = f"ffmpeg -y -i {temp_input} -ar 8000 -ac 1 -codec {codec} {temp_output}"
        if bitrate:
            command += f" -b:a {bitrate}"

        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        y_processed, _ = librosa.load(temp_output, sr=self.sr)
        os.remove(temp_input)
        os.remove(temp_output)
        return y_processed

    def save_audio(self, output_dir, filename, processed_signal):
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        sf.write(output_path, processed_signal, self.sr)
        print(f"Saved: {output_path}")