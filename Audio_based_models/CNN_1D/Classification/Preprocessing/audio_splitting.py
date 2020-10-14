import os

from scipy.io import wavfile
from spleeter.audio.adapter import get_default_audio_adapter
from spleeter.separator import Separator

from Audio_based_models.CNN_1D.Classification.Preprocessing.preprocessing_utils import load_wav_file


def separate_one_audio_on_accompaniment_and_vocals_by_spleeter(path_to_audio, sample_rate, output_directory):
    audio_loader = get_default_audio_adapter()
    separator = Separator('spleeter:2stems')
    filename=path_to_audio.split('/')[-1].split('\\')[-1]
    waveform, _ = audio_loader.load(path_to_audio, sample_rate=sample_rate)
    # Perform the separation :
    prediction = separator.separate(waveform)
    accompaniment=prediction['accompaniment']
    vocals=prediction['vocals']
    wavfile.write(output_directory + '.'.join(filename.split('.')[:-1])+'_accompaniment'+'.wav', sample_rate, accompaniment)
    wavfile.write(output_directory + '.'.join(filename.split('.')[:-1])+'_vocals'+'.wav', sample_rate, vocals)
    del audio_loader, separator, waveform, prediction, accompaniment, vocals
    gc.collect()

def separate_all_audios_on_accompaniment_and_vocals_by_spleeter(path_to_folder,path_to_destination_directory):
    filelist = os.listdir(path_to_folder)
    for file in filelist:
        data, audio_sample_rate = load_wav_file(path_to_folder + file)
        separate_one_audio_on_accompaniment_and_vocals_by_spleeter(path_to_folder + file, audio_sample_rate, path_to_destination_directory)