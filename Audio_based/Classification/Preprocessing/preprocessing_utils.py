import gc
import wave

import librosa
import moviepy.editor as mp
import os
import numpy as np
from scipy.io import wavfile


def extract_audio_from_video(path_to_video, path_to_save_extracted_audio):
    # Insert Local Video File Path
    clip = mp.VideoFileClip(path_to_video)
    # Insert Local Audio File Path
    clip.audio.write_audiofile(path_to_save_extracted_audio)

def extract_audios_from_videos_in_all_directory(path_to_directory_with_videos, path_to_destination_directory):
    filelist=os.listdir(path_to_directory_with_videos)
    for file in filelist:
        filename_for_audio='.'.join(file.split('.')[:-1])+'.wav'
        extract_audio_from_video(path_to_directory_with_videos+file, path_to_destination_directory+filename_for_audio)

def load_wav_file(path_to_file):
    frame_rate, data = wavfile.read(path_to_file)
    return data, frame_rate

def change_sample_rate_all_audios_in_folder(path_to_folder, needed_sample_rate, path_to_destination_directory):
    filelist = os.listdir(path_to_folder)
    for file in filelist:
        data, audio_sample_rate=load_wav_file(path_to_folder+file)
        del data
        data, audio_sample_rate=librosa.load(path_to_folder+file, audio_sample_rate)
        # resample
        data=librosa.resample(data, orig_sr=audio_sample_rate, target_sr=needed_sample_rate)
        librosa.output.write_wav(path_to_destination_directory+file, data, needed_sample_rate)


def extract_mfcc_from_audio(path_to_audio, n_fft,hop_length, n_mfcc, n_mels):
    sample_rate, f = wavfile.read(path_to_audio)
    y, sample_rate = librosa.load(path_to_audio, sr=sample_rate)
    mfcc_librosa = librosa.feature.mfcc(y=y, sr=sample_rate, n_fft=n_fft,
                                        n_mfcc=n_mfcc, n_mels=n_mels,
                                        hop_length=hop_length,
                                        fmin=0, fmax=None)
    return mfcc_librosa

def extract_mfcc_from_all_audios(path_to_dir_audio, path_to_output, n_fft,hop_length, n_mfcc, n_mels):
    audio_filenames=os.listdir(path_to_dir_audio)
    for filename in audio_filenames:
        mfcc=extract_mfcc_from_audio(path_to_dir_audio+filename, n_fft,hop_length, n_mfcc, n_mels)
        mfcc=np.transpose(mfcc)
        np.savetxt(path_to_output+filename.split('.')[0]+'.csv', mfcc, delimiter=',')


if __name__ == "__main__":
    # params
    path_to_video='D:\\Databases\\AffWild2\\Videos\\'
    path_of_extracted_audio='D:\\Databases\\AffWild2\\Extracted_audio\\'
    path_for_data_with_changed_sample_rate='D:\\Databases\\AffWild2\\Reduced_sample_rate\\'
    needed_sample_rate=16000
    # preprocessing
    #extract_audios_from_videos_in_all_directory(path_to_video, path_of_extracted_audio)
    #change_sample_rate_all_audios_in_folder(path_of_extracted_audio, needed_sample_rate, path_for_data_with_changed_sample_rate)

    # separation
    #output_directory='D:\\Databases\\AffWild2\\Separated_audios\\'
    #separate_all_audios_on_accompaniment_and_vocals_by_spleeter(path_for_data_with_changed_sample_rate, output_directory)

    # extraction MFCCs
    # params
    n_fft_params=[1600, 3200, 4800, 6400]
    n_mfcc_params=[13,23,30]
    n_mels_params=[32,64,128]
    for n_fft in n_fft_params:
        for n_mfcc in n_mfcc_params:
            for n_mels in n_mels_params:

                path_to_separated_audio='D:\\Databases\\AffWild2\\Separated_audios\\'
                path_to_output_audio='D:\\Databases\\AffWild2\\MFCC_features\\'
                #n_fft=3200
                hop_length=3200
                #n_mfcc=23
                #n_mels=128
                if not os.path.exists(path_to_output_audio+"mfcc_%i_n_fft_%i_hop_length_%i_n_mels_%i"%(n_mfcc, n_fft,hop_length,n_mels)+'\\'):
                    os.mkdir(path_to_output_audio+"mfcc_%i_n_fft_%i_hop_length_%i_n_mels_%i"%(n_mfcc, n_fft,hop_length,n_mels)+'\\')
                extract_mfcc_from_all_audios(path_to_dir_audio=path_to_separated_audio,
                                             path_to_output=path_to_output_audio+"mfcc_%i_n_fft_%i_hop_length_%i_n_mels_%i"%(n_mfcc, n_fft,hop_length,n_mels)+'\\',
                                             n_fft=n_fft,
                                             hop_length=hop_length,
                                             n_mfcc=n_mfcc,
                                             n_mels=n_mels)
