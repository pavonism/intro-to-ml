# Install packages: pip install noisereduce librosa soundfile

import os
import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np

# Folder path
main_folder = '/folder/with/wav/files'
output_folder = os.path.join(main_folder, 'denoised')

os.makedirs(output_folder, exist_ok=True)

for root, dirs, files in os.walk(main_folder):
    for file in files:
        if file.endswith('.wav'):

            file_path = os.path.join(root, file)
            y, sr = librosa.load(file_path, sr=None)

            # Reducing noise
            y_denoised = nr.reduce_noise(y=y, sr=sr)

            # Normalizing the audio
            if np.max(np.abs(y_denoised)) != 0:
                y_denoised = y_denoised / np.max(np.abs(y_denoised))

            # Resampling
            target_sr = 16000
            if sr != target_sr:
                y_denoised = librosa.resample(y_denoised, orig_sr=sr, target_sr=target_sr)
                sr = target_sr

            # Trim leading and trailing silence
            y_denoised, _ = librosa.effects.trim(y_denoised)

            # Saving the denoised audio file
            output_file_path = os.path.join(output_folder, file)
            sf.write(output_file_path, y_denoised, sr)

            print(f'Denoised audio saved to {output_file_path}')

print('Denoising completed for all .wav files.')