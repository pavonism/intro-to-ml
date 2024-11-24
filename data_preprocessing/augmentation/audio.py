import numpy as np
import audiomentations
import os
from tqdm import tqdm
from backend import audio_data
from backend.spectrograms.utils import AudioLoader
from backend.spectrograms import utils
from backend.tsrc.data import Environments
from data_preprocessing.augmentation.compose import Compose
from math import ceil
import soundfile as sf
import shutil
import random
import librosa
from pedalboard import Pedalboard, Reverb


def addReverb(audio, room_size=0.2, damping=0.7, wet_level=0.15, dry_level=0.85):
    board = Pedalboard([
    Reverb(
        room_size=room_size,    # 0 to 1
        damping=damping,      # 0 to 1
        wet_level=wet_level,   # 0 to 1
        dry_level=dry_level,    # 0 to 1
    )
    ])
    audio.samples = board.process(audio.samples, audio.sample_rate)
    return audio

def getFirstSyllable(audio):
    _audio = copy.deepcopy(audio)
    
    _, _interval = librosa.effects.trim(_audio.samples, top_db=15)
    _interval[0] = max(_interval[0] - 40, 0)
    _audio.samples = _audio.samples[_interval[0]:_interval[1]]
    cutoff_point = int(len(_audio.samples) * 5/10)
    _audio.samples = _audio.samples[:cutoff_point]
    return _audio

def addNoise(audio : audio_data.AudioData, noiseThreshold):
        y = audio.samples
        noise = np.random.normal(0, noiseThreshold, len(y))
        audio.samples = y + noise
        return audio

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

def change_pitch(audio : audio_data.AudioData, n_steps=[-2, -1, 1, 2]):
    """
    Change pitch of audio
    y: audio signal
    sr: sample rate
    n_steps: array cointaining number of steps to shift pitch. change_pitch() chooses one random value from array.
    """
    n_steps_choice = np.random.choice(n_steps)
    audio.samples = librosa.effects.pitch_shift(audio.samples, sr=audio.sample_rate, n_steps=n_steps_choice)
    return audio

class AugmentAudio:
    def __init__(self, input_path):
        self.input_path = input_path
        print(f"Loading audio files... {os.path.join(input_path, 'train')}")
        self.audioDataTrain = AudioLoader().load_data(os.path.join(input_path, 'train'))
        print(f"Loading audio files... {os.path.join(input_path, 'validation')}")
        self.audioDataValidation = AudioLoader().load_data(os.path.join(input_path, 'validation'))

    def getClassRecordings(self, dataset_name, env):
        if dataset_name == 'train':
            audioList = self.audioDataTrain
        if dataset_name == 'validation':
            audioList = self.audioDataValidation

        result = []
        for audio in audioList:
            if audio.class_id == env:
                result.append(audio)
        print(f"env:{env} COUNT: {len(result)}")
        return result

    def getUniqueSpeakers(self, env):
        a = self.audioDataTrain[0]
        speakersSet=set()
        for audio in self.audioDataTrain:
            if audio.class_id == env:
                speakersSet.add(audio)
        return list(speakersSet)
    
    def getSpeakerRecordings(self, audio_input, env):
        audioSet=set()
        speaker_id_search = audio_input.speaker_id.split('_')[0]
        for audio in self.audioDataTrain:
            speaker_id_found = audio.speaker_id.split('_')[0]
            #print(f"env : {audio.class_id} - {env}, speaker_id: {audio.speaker_id} - {audio_input.speaker_id}")
            if audio.class_id == env and speaker_id_search == speaker_id_found:
                #print("FOUND")
                audioSet.add(audio)
        return list(audioSet)
    
    def augmentWithRatio(self, audio, outputPath : str, augmentationRatio):
        """
        Performs number of augmentations basing on augmentationRatio
        """
        _audio = audio

    def augmentAndSaveAudioFile(self, audio : audio_data.AudioData, compose_pipeline : Compose, pathToFolder : str):
        """
        First augments audio file than saves in pathToFolder with pattern:
        speakerId_augmentationInfoString.wav
        """
        augmented_recording, infoString = compose_pipeline(audio, minOneAugmentation=True)
        file_name = augmented_recording.speaker_id.removesuffix(".wav")
        pathToAugmented = self.joinAndNormPath(pathToFolder, file_name + '_' + str(infoString) + ".wav")
        sf.write(pathToAugmented, augmented_recording.samples, augmented_recording.sample_rate)

    def augmentAndSaveAudioFileMultiple(self, audio, compose_pipeline : Compose, pathToFolder : str, ratio : float):
        """
        Augmnets audio based on ratio.

        Methodology
        -----------
        If ratio > 1
            augment audio untill ratio less than 1
            ratio--
        If 0 < ratio < 1
            if random float from (0, 1) < ratio:
                augment audio

        """
        count = ratio
        while(count >0):
            if count >= 1:
                self.augmentAndSaveAudioFile(audio, compose_pipeline, pathToFolder)
                count = count - 1

            if count > 0 and count < 1:
                if random.uniform(0, 1) < count:
                    self.augmentAndSaveAudioFile(audio, compose_pipeline, pathToFolder)
                count = count -1
    
    @staticmethod
    def joinAndNormPath(*paths : list):
        path = os.path.join(*paths)
        return os.path.normpath(path)


    
    def augmentDataset(self, input_path : str, output_path : str, compose_pipeline : Compose, wantedNoFilesPerClass, fileTag):
        assert wantedNoFilesPerClass is not None
        assert utils.AudioLoader.split_path(input_path, -1) in ['train', 'validation']

        dataset_name = utils.AudioLoader.split_path(input_path, -1)

        print(f"Augmenting {dataset_name}")
        for env in Environments.get_all_clean():
            filesInEnv = self.getClassRecordings(dataset_name, env)
            if len(filesInEnv) == 0:
                print(f"No files found in loaded dataset for env: {env}")
                return

            noFilesToAugment = wantedNoFilesPerClass - len(filesInEnv)
            augmentationRatio = noFilesToAugment/len(filesInEnv)

            if augmentationRatio < 0:
                print(f"WARNING: agumentationRatio less than 0: {augmentationRatio}\nNo need for data augmentation in env: {env}")
                continue
            
            print(f"wantedNoFilesPerClass : {wantedNoFilesPerClass}, filesInEnv: {len(filesInEnv)}, augmentationRatio: {augmentationRatio}")
            for recording in filesInEnv:
                # saving original oudio
                os.makedirs(os.path.join(output_path, env) , exist_ok=True)
                pathToFolder = os.path.join(output_path, env)
                pathToNotModified = self.joinAndNormPath(pathToFolder, recording.speaker_id.removesuffix(".wav") + '_original' + '.wav')
                sf.write(pathToNotModified, recording.samples, recording.sample_rate)

                # augment recording based on augmentationRatio
                self.augmentAndSaveAudioFileMultiple(recording, compose_pipeline, pathToFolder, augmentationRatio)
                    
                    
    
    def splitAugmentation(self, input_path: str, output_path : str, compose_pipeline : Compose, wantedNoFilesPerClassTrain :int, wantedNoFilesPerClassVal : int, fileTag = ""):
        """
        Makes sure that augmentation is only done on train.

        Copies val and test to new location
        """
        #Augmenting train
        _output_path = os.path.join(output_path, "train")
        train_path = os.path.join(input_path, "train")
        self.augmentDataset(train_path, _output_path, compose_pipeline, wantedNoFilesPerClassTrain, fileTag)

        #Augmenting validation
        _output_path = os.path.join(output_path, "validation")
        validation_path = os.path.join(input_path, "validation")
        self.augmentDataset(validation_path, _output_path, compose_pipeline, wantedNoFilesPerClassVal, fileTag)

        print("Copying over test set")
        for root, dirs, files in tqdm(os.walk(input_path)):
            train_test_val = utils.AudioLoader.split_path(root, -2)
            if train_test_val in ["test"]:
                for file in files:
                    class_id = utils.AudioLoader.split_path(root, -1)
                    # if no directory create one
                    _output_path = os.path.join(output_path, train_test_val, class_id)
                    os.makedirs(_output_path, exist_ok=True)
                    shutil.copy2(os.path.join(root, file), os.path.join(_output_path, file))
            else:
                #print(f"Skipping directory {dirs}, root {root}, split {train_test_val}")
                ...
        
        

    
    def __call__(self, output_path: str, compose_pipeline : Compose, wantedNoFilesPerClassTrain, wantedNoFilesPerClassVal, fileTag):
        self.splitAugmentation(self.input_path, output_path, compose_pipeline, wantedNoFilesPerClassTrain, wantedNoFilesPerClassVal, fileTag)
        

def augment_audio_files(input_path: str, output_path: str, transformation, wantedNoFilesPerClassTrain, wantedNoFilesPerClassVal, fileTag = ""):
    AugmentAudio(input_path)(output_path, transformation, wantedNoFilesPerClassTrain, wantedNoFilesPerClassVal, fileTag)
