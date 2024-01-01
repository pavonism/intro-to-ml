import numpy as np
import audiomentations
import os
import tqdm
from backend.spectrograms.utils import AudioLoader
from backend.tsrc.data import Environments
from data_preprocessing.augmentation.compose import Compose
from math import ceil
import soundfile as sf
import shutil

def addNoise(y, sr, noiseThreshold):
        noise = np.random.normal(0, noiseThreshold, len(y))
        return y + noise, sr 
    
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
        self.audioDataTrain = AudioLoader().load_data(os.path.join(input_path, "/Train/"))

    def getUniqueSpeakers(self, env):
        speakersList=set()
        for audio in self.audioDataTrain:
            if audio.class_id == env:
                speakersList.add(audio.speaker_id)
        return speakersList
    
    def getSpeakerRecordings(self, speaker_id, env):
        audioList=[]
        for audio in self.audioDataTrain:
            if audio.class_id == env and audio.speaker_id == speaker_id:
                audioList.add(audio)
        return audioList
    
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
                    pathToSave = output_path + recording.train_validaton_test + f"/{env}/" + augmented_recording.speaker_id + infoString + r".wav"
                    pathToSave = os.path.normpath(pathToSave)
                    sf.write(pathToSave, augmented_recording.samples, augmented_recording.sample_rate)
    
    def splitAugmentation(self, input_path: str, output_path : str, compose_pipeline : Compose, wantedNoFilesPerSpeaker):
        """
        Makes sure that augmentation is only done on train.

        Copies val and test to new location
        """
        for root, dirs, files in os.walk(input_path):
            if files is None:
                continue
            print(f"root: {root}, dirs: {dirs}, files[0] {files[0]}")
            return
            if dirs == "Train":
                train_path = os.path.join(input_path, "/Train/")
                self.augmentTrain(train_path, output_path, compose_pipeline, wantedNoFilesPerSpeaker)
            elif dirs in ["Test", "Validation"]:
                for file in files:
                    # if no directory create one
                    _output_path = os.path.join(output_path, dirs)
                    os.makedirs(_output_path, exist_ok=True)
                    shutil.copy2(os.path.join(root, dirs, file), os.path.join(output_path, file))
            else:
                print(f"Skipping directory {dirs}")

    
    def __call__(self, input_path: str, output_path: str, compose_pipeline : Compose, wantedNoFilesPerSpeaker):
        self.splitAugmentation(input_path, output_path, compose_pipeline, wantedNoFilesPerSpeaker)
        
                



def augment_audio_files(input_path: str, output_path: str, transformation):
    print(f"Loading audio files... {os.path.abspath(input_path)}")
    audioData = AudioLoader().load_data(input_path)


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
