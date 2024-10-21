import audio_to_spectrogram_converter
import glob
import os

class1_speakers = ['f1', 'f7', 'f8', 'm3', 'm6', 'm8']

def If_Class1(file_name : str):
    speaker = (file_name.split('_'))[0]
    if(speaker == '.'):
        speaker = (file_name.split('_'))[1]
    for el in class1_speakers:
        if(el == speaker):
            return "Class1"
    return "Class0"

def Create_Dirs(image_path):
        if(os.path.exists(image_path) == False):
            os.makedirs(image_path)
        if(os.path.exists(image_path + "/Class1") == False):
            os.makedirs(image_path + "/Class1")
        if(os.path.exists(image_path + "/Class0") == False):
            os.makedirs(image_path + "/Class0")


def SpectsLoader(audio_path , image_path):

    Create_Dirs(image_path)
    con = audio_to_spectrogram_converter.AudioToSpectrogramConverter()

    found_files = glob.glob(f"{audio_path}/**.wav", recursive=True)

    for file in found_files:
        file_name = (((file.split("\\"))[1]).split('.'))[0]
        clas = If_Class1(file_name)
        if(os.path.exists(image_path + f"/{clas}/" + file_name + ".png") == False):
            con.convert_file(file , image_path + f"/{clas}/" + file_name + ".png")
