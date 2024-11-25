import os

class SpectrogramData():
    def __init__(self, file_path : str, speaker_id : str, recording_no : int , train_validaton_test : str, class_id : int):
        self.file_path = file_path
        self.speaker_id = speaker_id
        self.recording_no = recording_no
        self.train_validaton_test = train_validaton_test
        self.class_id = class_id

class SpectrogramLoader():
        
    @staticmethod
    def load_data(path):
        objects_list = []
        
        for subdir, dirs, files in os.walk(path):
            if dirs is not None:
                for file in files:
                    file_path = os.path.join(subdir, file)
                    file_path = os.path.normpath(file_path)
                    class_id = file_path.split('\\')[-2]
                    train_validaton_test = file_path.split('\\')[-3]
                    speaker_id = file[:8]
                    recording_no = file.removesuffix(".png").split("_")[1]

                    data_object = SpectrogramData(file_path=file_path,
                                                  speaker_id=speaker_id,
                                                  recording_no=int(recording_no),
                                                  train_validaton_test=train_validaton_test,
                                                  class_id=int(class_id))
                    objects_list.append(data_object)
        
        return objects_list
