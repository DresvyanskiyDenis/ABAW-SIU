import os
import numpy as np
import tensorflow as tf
import pandas as pd

from Audio_based.Classification.Preprocessing.labels_utils import transform_probabilities_to_original_sample_rate
from Audio_based.utils.Database_instance import Database_instance
from Audio_based.utils.Generator_audio import predict_data_with_the_model
from Audio_based.utils.models import CNN_1D_model
from Audio_based.utils.utils import load_data_wav


def generate_test_predictions(path_to_data, labels_filename, model, model_output_sample_rate, path_to_video, window_size, window_step, prediction_mode):

    instance = Database_instance()
    instance.loading_data_function = load_data_wav
    instance.load_data(path_to_data.split('_left')[0].split('_right')[0].split('_vocals')[0].split('.')[0] +'_vocals.'+path_to_data.split('.')[-1])
    instance.label_filename = labels_filename
    instance.labels, instance.labels_frame_rate = np.array([[0],[0]]), model_output_sample_rate
    instance.align_number_of_labels_and_data()
    instance.generate_timesteps_for_labels()
    instance.cut_data_and_labels_on_windows(window_size, window_step)
    predict_data_with_the_model(model, [instance], prediction_mode=prediction_mode)
    dict_filename_to_predictions = transform_probabilities_to_original_sample_rate(
        database_instances=[instance],
        path_to_video=path_to_video,
        original_sample_rate=model_output_sample_rate,
        need_save=False)
    return dict_filename_to_predictions

def generate_test_predictions_from_list(list_filenames, path_to_data, model, model_output_sample_rate, path_to_video,
                                        window_size, window_step, path_to_output,prediction_mode):
    for filename in list_filenames:
        path_to_audio=path_to_data+filename.split('.')[0]+'_vocals.wav'
        tmp_dict=generate_test_predictions(path_to_audio,filename, model, model_output_sample_rate, path_to_video, window_size, window_step, prediction_mode)
        tmp_dict[filename+'.csv'].to_csv(path_to_output+filename.split('.')[0]+'.csv', header=False, index=False)

if __name__ == "__main__":
    path_to_filenames_labels='C:\\Users\\Dresvyanskiy\\Desktop\\expression_test_set.txt'
    filenames=pd.read_csv(path_to_filenames_labels, header=None).values
    #filenames=np.array(os.listdir('D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\validation\\Aligned_labels_reduced\\sample_rate_5\\'))
    filenames=filenames.reshape((-1,))
    path_to_weights='C:\\Users\\Dresvyanskiy\\Downloads\\best_model_weights_1D_CNN.h5'
    path_to_data='D:\\Databases\\AffWild2\\Separated_audios\\'
    path_to_output= '../../logs/predictions_test\\'
    path_to_video='D:\\Databases\\AffWild2\\Videos\\'
    if not os.path.exists(path_to_output):
        os.mkdir(path_to_output)
    window_size=4
    window_step=window_size/5.*2.
    num_classes = 7
    prediction_mode='sequence_to_one'
    model_output_sample_rate=5
    # model params
    model_input = (window_size*16000,1)
    optimizer = tf.keras.optimizers.Nadam()
    loss = tf.keras.losses.categorical_crossentropy
    # create model
    model = CNN_1D_model(model_input, num_classes)
    model.load_weights(path_to_weights)
    if prediction_mode == 'sequence_to_sequence':
        model.compile(optimizer=optimizer, loss=loss, sample_weight_mode="temporal")
    else:
        model.compile(optimizer=optimizer, loss=loss)

    generate_test_predictions_from_list(list_filenames=filenames,
                                        path_to_data=path_to_data,
                                        model=model, model_output_sample_rate=model_output_sample_rate,
                                        path_to_video=path_to_video,
                                        window_size=window_size, window_step=window_step,
                                        path_to_output=path_to_output,
                                        prediction_mode=prediction_mode)