# import torchaudio


# def convert_audio_to_16k_mono(input_path, output_path):
#     waveform, sample_rate = torchaudio.load(input_path)

#     if waveform.shape[0] > 1:
#         waveform = waveform.mean(dim=0, keepdim=True)

#     if sample_rate != 16000:
#         resampler = torchaudio.transforms.Resample(
#             orig_freq=sample_rate,
#             new_freq=16000
#         )
#         waveform = resampler(waveform)

#     torchaudio.save(output_path, waveform, 16000)
import os
os.environ["PATH"] += os.pathsep + r"D:\SHIVANI\INTERNSHIP\CDAC\cheat\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin"

from pydub import AudioSegment
AudioSegment.converter = r"D:\SHIVANI\INTERNSHIP\CDAC\cheat\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"

import librosa
import torchaudio
import torch


def convert_audio_to_16k_mono(input_path, output_path):
    try:
        # try normal way
        waveform, sample_rate = torchaudio.load(input_path)

    except Exception:
        # fallback for browser recordings
        y, sample_rate = librosa.load(input_path, sr=None, mono=True)
        waveform = torch.tensor(y).unsqueeze(0)

    # mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000
        )
        waveform = resampler(waveform)

    torchaudio.save(output_path, waveform, 16000)
