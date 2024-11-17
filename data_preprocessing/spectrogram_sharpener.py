import glob
import os
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
from tqdm import tqdm


def sharpen_spectrogram(spectrogram: Image.Image, sigma=1, alpha=1.5):
    """
    Sharpens a spectrogram by applying a Gaussian filter and enhancing the edges.

    Parameters:
    spectrogram (PIL.Image.Image): Input spectrogram to be sharpened.
    sigma (float): Standard deviation for Gaussian kernel.
    alpha (float): Sharpening factor.

    Returns:
    PIL.Image.Image: Sharpened spectrogram.
    """

    spectrogram_array = np.array(spectrogram).astype(np.float32)
    blurred = gaussian_filter(spectrogram_array, sigma=sigma)
    sharpened_array = spectrogram_array + alpha * (spectrogram_array - blurred)
    sharpened_array = np.clip(sharpened_array, 0, 255)
    sharpened = Image.fromarray(sharpened_array.astype(np.uint8))

    return sharpened


def sharpen_spectrograms(input_path: str, output_path: str):
    for subset in ["train", "test", "validation"]:
        for word in os.listdir(f"{input_path}/{subset}"):
            if word == "_background_noise_":
                continue

            word_directory = f"{output_path}/{subset}/{word}"
            os.makedirs(word_directory, exist_ok=True)

            for file in tqdm(
                glob.glob(
                    f"{input_path}/{subset}/{word}/*.png",
                    recursive=True,
                ),
                desc=f"Dataset: {subset}. Converting {word} files to images",
            ):
                output_file = f"{word_directory}/{Path(file).stem}.png"

                if os.path.exists(output_file):
                    continue

                sharpened_spectrogram = sharpen_spectrogram(Image.open(file))
                sharpened_spectrogram.save(output_file)
