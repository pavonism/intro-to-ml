from backend import audio_data

from tqdm import tqdm
import os
import librosa

class AudioLoader():
        
    @staticmethod
    def __load_audio(file_path: str, train_validaton_test : str, speaker_id : str, class_id : str) -> audio_data.AudioData:
        y, sr = librosa.load(file_path)
        return audio_data.AudioData(y, sr, train_validaton_test, speaker_id, class_id)
    
    @staticmethod
    def split_path(path, index):
        try:
            path = os.path.normpath(path)
        except:
            print(f"Cannot norm path {path}")

        if len(path.split('/')) > 1:
            return path.split('/')[index]
        elif len(path.split('\\')) > 1:
            return path.split('\\')[index]
        else:
            raise Exception("Unable to split path {path}")
    
    @staticmethod
    def load_data(path):
        objects_list = []
        
        for subdir, dirs, files in os.walk(path):
            if dirs is not None:
                for file in tqdm(files):
                    file_path = os.path.join(subdir, file)
                    file_path = os.path.normpath(file_path)
                    train_validaton_test = AudioLoader.split_path(file_path, -3)
                    speaker_id = AudioLoader.split_path(file_path, -1)
                    class_id = AudioLoader.split_path(file_path, -2)
                    #print(f"train_test : {train_validaton_test}, speaker_id: {speaker_id}, class_id : {class_id}")

                    data_object = AudioLoader.__load_audio(file_path=file_path,
                                                  train_validaton_test=train_validaton_test,
                                                  speaker_id=speaker_id,
                                                  class_id=class_id)
                    objects_list.append(data_object)
        
        return objects_list
