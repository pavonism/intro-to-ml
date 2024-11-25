from backend.spectrograms import utils

import os
import random
import cv2
import numpy as np

import audiomentations
import librosa


random.seed(420)

def image_crop(image, range_col):
    """
    Crops image to given range_col
    
    Example
    -------
    # crop image to x = [100, 200]
    >>> image_crop(image, [100, 200])
    """
    print(f"range_col: {range_col}")
    return image[:, range_col[0] : range_col[1]]

def image_strech(image, col_pixels = 300):
    """
    Stretches image to given width
    
    Examples
    --------
    image # image of height 400 and width 145
    >>> image = image_stretch(image, 300)
    image # image of height 400 and width 300
    """
    return cv2.resize(image, (col_pixels, 400))

def image_random_crop(image, col_percentage = 0.5):
    pixels_cropped = int(300*col_percentage)
    start_col = random.randint(0,300-pixels_cropped)
    return image_crop(image, [start_col, start_col + pixels_cropped])

def image_random_crop_and_resize(image, col_percentage = 0.5):
    transformed_image = image_random_crop(image, col_percentage=col_percentage)
    return image_strech(transformed_image)


def spec_augment(spec: np.ndarray, num_mask=2, 
                 freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):

    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
        
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)
        
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0
    
    return spec






