import numpy as np
import audiomentations
import os
from tqdm import tqdm
from backend import audio_data
from backend.spectrograms.utils import AudioLoader
from backend.tsrc.data import Environments
from data_preprocessing.augmentation.compose import Compose
from math import ceil
import soundfile as sf
import shutil

def addNoise(audio : audio_data.AudioData, noiseThreshold):
        y = audio.samples
        noise = np.random.normal(0, noiseThreshold, len(y))
        audio.samples = y + noise
        return audio
    
def addReverb(y, sr):
    """
    Add reverberation effect using audiomentations with correct parameters
    """
    reverb = audiomentations.RoomSimulator(
        min_size_x=3.6,  # Minimum room width
        max_size_x=5.6,  # Maximum room width
        min_size_y=3.6,  # Minimum room depth
        max_size_y=3.9,  # Maximum room depth
        min_size_z=2.4,  # Minimum room height
        max_size_z=3.0,  # Maximum room height
        min_absorption_value=0.075,  # Minimum surface absorption
        max_absorption_value=0.4,    # Maximum surface absorption
        calculation_mode="absorption",  # Use absorption-based calculation
        use_ray_tracing=True,        # Enable ray tracing for accuracy
        max_order=1,                 # Maximum order of reflections
        p=1.0                        # Probability of applying the effect
    )
    return reverb(samples=y, sample_rate=sr)

def shift_and_pad(y, sr, shift_factor = 0.5):
    """
    Shift audio and pad with silence
    y: audio signal
    shift_samples: number of samples to shift
    """
    shift_samples = int(y.shape[0] * shift_factor)
    shifted = np.roll(y, shift_samples)
    if shift_samples > 0:
        shifted[:shift_samples] = 0
    else:
        shifted[shift_samples:] = 0
    return shifted, sr

def change_pitch(y, sr, n_steps=4):
    """
    Change pitch of audio
    y: audio signal
    sr: sample rate
    n_steps: number of steps to shift pitch
    """
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def add_reverb(y, sr):
    """
    Add reverberation effect using audiomentations with correct parameters
    """
    reverb = audiomentations.RoomSimulator(
        min_size_x=3.6,  # Minimum room width
        max_size_x=5.6,  # Maximum room width
        min_size_y=3.6,  # Minimum room depth
        max_size_y=3.9,  # Maximum room depth
        min_size_z=2.4,  # Minimum room height
        max_size_z=3.0,  # Maximum room height
        min_absorption_value=0.075,  # Minimum surface absorption
        max_absorption_value=0.4,    # Maximum surface absorption
        calculation_mode="absorption",  # Use absorption-based calculation
        use_ray_tracing=True,        # Enable ray tracing for accuracy
        max_order=1,                 # Maximum order of reflections
        p=1.0                        # Probability of applying the effect
    )
    return reverb(samples=y, sample_rate=sr)

class AugmentAudio:
    def __init__(self, input_path):
        self.input_path = input_path
        print(f"Loading audio files... {os.path.join(input_path, 'train', 'bed')}")
        self.audioDataTrain = AudioLoader().load_data(os.path.join(input_path, 'train', 'bed'))

    def getUniqueSpeakers(self, env):
        speakersSet=set()
        for audio in self.audioDataTrain:
            if audio.class_id == env:
                speakersSet.add(audio.speaker_id)
        return list(speakersSet)
    
    def getSpeakerRecordings(self, speaker_id, env):
        audioSet=set()
        for audio in self.audioDataTrain:
            if audio.class_id == env and audio.speaker_id == speaker_id:
                audioSet.add(audio)
        return list(audioSet)
    
    def augmentTrain(self, input_path : str, output_path : str, compose_pipeline : Compose, wantedNoFilesPerSpeaker):
        for env in Environments.get_all_clean():
            for speaker in self.getUniqueSpeakers(env):
                speakerRecordings = self.getSpeakerRecordings(speaker, env)
                noRecordings = len(speakerRecordings)
                wantedAugmentationsPerFile = wantedNoFilesPerSpeaker/noRecordings
                if wantedNoFilesPerSpeaker - noRecordings < 0:
                    raise Exception(f"wantedNoFilesPerSpeaker == {wantedNoFilesPerSpeaker} but in base dataset there are already {noRecordings} number of recordings for env {env} and speaker {speaker}")
                for recording in speakerRecordings:
                    augmented_recording, infoString = compose_pipeline(recording, ceil(wantedAugmentationsPerFile))
                    file_name = augmented_recording.file_path.split('//')[-1].removesuffix(".wav")
                    pathToSave = os.path.join(output_path, env, file_name, infoString, ".wav")
                    pathToSave = os.path.normpath(pathToSave)
                    print(f"Saving... in path {pathToSave}")
                    return
                    sf.write(pathToSave, augmented_recording.samples, augmented_recording.sample_rate)
    
    def splitAugmentation(self, input_path: str, output_path : str, compose_pipeline : Compose, wantedNoFilesPerSpeaker):
        """
        Makes sure that augmentation is only done on train.

        Copies val and test to new location
        """
        for root, dirs, files in tqdm(os.walk(input_path)):
            train_test_val = os.path.normpath(root).split('\\')[-2]
            if train_test_val in ["test", "validation"]:
                for file in files:
                    # if no directory create one
                    _output_path = os.path.join(output_path, train_test_val)
                    os.makedirs(_output_path, exist_ok=True)
                    shutil.copy2(os.path.join(root, file), os.path.join(_output_path, file))
            else:
                print(f"Skipping directory {dirs}, root {root}, split {train_test_val}")
        
        #Augmenting train
        _output_path = os.path.join(output_path, train_test_val)
        train_path = os.path.join(input_path, "train")
        self.augmentTrain(train_path, _output_path, compose_pipeline, wantedNoFilesPerSpeaker)

    
    def __call__(self, output_path: str, compose_pipeline : Compose, wantedNoFilesPerSpeaker):
        self.splitAugmentation(self.input_path, output_path, compose_pipeline, wantedNoFilesPerSpeaker)
        
                



def augment_audio_files(input_path: str, output_path: str, transformation, wantedNoFilesPerSpeaker):
    AugmentAudio(input_path)(output_path, transformation, wantedNoFilesPerSpeaker)


    #for env in Environments.get_all_clean():
    #    get_unique_speakers(env)
    #    for speakers in get_unique_speakers():



    return
    for root, subdir, files in os.walk(input_path):
        test_val_split = os.path.normpath(root).split('\\')[-2]
        print(f"root: {root}, dirs : {subdir}, files: {files}, type: {test_val_split}")
        if subdir =="Train":
            for file in tqdm(files):
                file_path = os.path.join(subdir, file)
                file_path = os.path.normpath(file_path)
                train_validaton_test = file_path.split('\\')[-3]
                speaker_id = file.split('_')[1]
                class_id = file_path.split('\\')[-2]


#if __name__=="__main__":
#    augment_audio_files(r"./data/v3/tsrc_train_test", './data/v3/data_augmented')
