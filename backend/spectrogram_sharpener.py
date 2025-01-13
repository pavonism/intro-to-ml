import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image


class SpectrogramSharpener:
    def sharpen_spectrogram(
        self,
        spectrogram: Image.Image,
        sigma=1,
        alpha=1.5,
    ) -> Image.Image:
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

    def sharpen_spectrogram_file(
        self,
        input_file: str,
        output_file: str,
        sigma=1,
        alpha=1.5,
    ):
        """
        Sharpens a spectrogram file and saves the result to a new file.

        Parameters:
        input_file (str): Path to the input spectrogram file.
        output_file (str): Path to the output file.
        sigma (float): Standard deviation for Gaussian kernel.
        alpha (float): Sharpening factor.
        """

        spectrogram = Image.open(input_file)
        sharpened = self.sharpen_spectrogram(spectrogram, sigma=sigma, alpha=alpha)
        sharpened.save(output_file)
