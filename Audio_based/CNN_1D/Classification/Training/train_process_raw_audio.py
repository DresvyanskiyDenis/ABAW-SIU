import os
import pandas as pd
import numpy as np
from sklearn.utils import class_weight

from Audio_based.CNN_1D.utils.Metric_calculator import Metric_calculator
from Audio_based.CNN_1D.utils.models import CNN_1D_model
from Audio_based.CNN_1D.utils.Database import Database
from Audio_based.CNN_1D.utils.Generator_audio import batch_generator_cut_data, predict_data_with_the_model
from Audio_based.CNN_1D.utils.utils import load_labels, load_data_wav, generate_weights, find_the_greatest_class_in_array
import tensorflow as tf



def train_model_on_data(path_to_data, path_to_labels_train, path_to_labels_validation, path_to_output, window_size, window_step,
                        class_weights_mode='my_realisation', prediction_mode='sequence_to_sequence', save_model_every_batch=False,
                        load_weights_before_training=False, path_to_weights=None, validation_value_best_model=None):
    # data params
    path_to_data_train = path_to_data
    path_to_labels_train = path_to_labels_train

    train_database = Database(path_to_data=path_to_data_train,
                              path_to_labels=path_to_labels_train,
                              data_filetype='wav',
                              data_postfix='_vocals')
    train_database.load_all_data_and_labels(loading_data_function=load_data_wav, loading_labels_function=load_labels)
    train_database.prepare_data_for_training(window_size=window_size, window_step=window_step,
                                                               need_scaling=False,
                                                               scaler=None,
                                                               return_scaler=False)

    # validation data
    validation_database = Database(path_to_data=path_to_data_train,
                              path_to_labels=path_to_labels_validation,
                              data_filetype='wav',
                              data_postfix='_vocals')
    validation_database.load_all_data_and_labels(loading_data_function=load_data_wav, loading_labels_function=load_labels)
    validation_database.prepare_data_for_training(window_size=window_size, window_step=window_step,
                                                                 delete_value=None,
                                                                 need_scaling=False,
                                                                 scaler=None,
                                                                 return_scaler=False)




    # model params
    model_input=(train_database.data_instances[0].data_window_size,)+train_database.data_instances[0].data.shape[1:]
    num_classes=7
    batch_size=20
    epochs=2
    optimizer=tf.keras.optimizers.Nadam()
    loss=tf.keras.losses.categorical_crossentropy
    # create model
    model= CNN_1D_model(model_input, num_classes)

    if load_weights_before_training:
        model.load_weights(path_to_weights)

    if prediction_mode == 'sequence_to_sequence':
        model.compile(optimizer=optimizer, loss=loss, sample_weight_mode="temporal")
    else:
        model.compile(optimizer=optimizer, loss=loss)



    # class weighting through sample weighting, while keras do not allow use class_weights with reccurent layers and 3D+ data
    if class_weights_mode == 'my_realisation':
        class_weights = generate_weights(
            np.unique(train_database.get_all_concatenated_data_and_labels()[1], return_counts=True)[1])
    elif class_weights_mode == 'scikit':
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(
                                                              train_database.get_all_concatenated_data_and_labels()[1]),
                                                          train_database.get_all_concatenated_data_and_labels()[1])

    # calculate metric on validation

    best_result=0
    if validation_value_best_model!=None:
        best_result=validation_value_best_model

    for epoch in range(epochs):
        train_generator = batch_generator_cut_data(train_database.data_instances, need_shuffle=True,
                                                   batch_size=batch_size, need_sample_weight=True, class_weights=class_weights)
        num_batch=0
        loss_sum=0
        for generator_step in train_generator:
            train_data, train_labels, sample_weights=generator_step
            # if we have want to predict only one labels per whole window, we need to reduce all labels in window to one,
            # which is the majority of window with labels
            if prediction_mode=='sequence_to_one':
                new_labels=np.zeros(shape=(train_labels.shape[0], 1))
                for i in range(train_labels.shape[0]):
                    greatest_class=find_the_greatest_class_in_array(train_labels[i])
                    new_labels[i,0]=greatest_class
                train_labels=new_labels

            train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
            train_data, train_labels, sample_weights = train_data.astype('float32'), train_labels.astype('float32'), sample_weights.astype('float32')
            if prediction_mode=='sequence_to_one':
                train_result = model.train_on_batch(train_data, train_labels, class_weight=class_weights)
            elif prediction_mode=='sequene_to_sequence':
                train_result=model.train_on_batch(train_data, train_labels, sample_weight=sample_weights)
            #print('Epoch %i, num batch:%i, loss:%f'%(epoch, num_batch,train_result))
            num_batch+=1
            loss_sum+=train_result
        # calculate metric on validation with extension of labels to original sample rate (videos frame rate)
        predict_data_with_the_model(model, validation_database.data_instances, prediction_mode=prediction_mode)

        validation_result = Metric_calculator(None, None,None).\
            calculate_FG_2020_F1_and_accuracy_scores_with_extended_predictions(instances=validation_database.data_instances,
                                                                               path_to_video='',
                                                                               path_to_real_labels='',
                                                                               original_sample_rate=5,
                                                                               delete_value=-1)
        print('Epoch %i is ended. Average loss:%f, validation FG-2020 metric:%f' % (epoch, loss_sum / num_batch, validation_result))
        if validation_result>=best_result:
            best_result=validation_result
            model.save_weights(path_to_output+'best_model_weights.h5')
            results=pd.DataFrame(columns=['data directory', 'window size', 'validation_result'])
            results=results.append({'data directory':path_to_data, 'window size':window_size, 'validation_result':best_result}, ignore_index=True)
            results.to_csv(path_to_output+'test_results.csv', index=False)
        if save_model_every_batch:
            model.save_weights(path_to_output + 'last_epoch_model_weights.h5')
    return best_result



if __name__ == "__main__":
    # data params
    path_to_data='D:\\Databases\\AffWild2\\Separated_audios\\'
    path_to_labels_train='D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\train\\dropped14_interpolated10\\'
    path_to_labels_validation = 'D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\validation\\Aligned_labels_reduced\\sample_rate_5\\'
    path_to_output= 'results\\'

    if not os.path.exists(path_to_output):
        os.mkdir(path_to_output)
    #data_directories=os.listdir(path_to_data)

    window_sizes=[4]

    results=pd.DataFrame(columns=['data directory', 'window size', 'validation_result'])
    class_weights_mode='scikit'
    prediction_mode='sequence_to_one'
    save_model_every_batch=True
    load_weights_before_training=True
    need_load_result_best_model=True
    for window_size in window_sizes:
        output_directory=path_to_output+'1D_CNN_window_size_'+str(window_size)+'\\'
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        if need_load_result_best_model:
            res_best_model=pd.read_csv(output_directory+'test_results.csv')
            best_value=res_best_model['validation_result'].values[0]
        val_result=train_model_on_data(path_to_data=path_to_data,
                                       path_to_labels_train=path_to_labels_train,
                                       path_to_labels_validation=path_to_labels_validation,
                                       path_to_output=output_directory,
                                       window_size=window_size,
                                       window_step=window_size*2./5.,
                                       class_weights_mode=class_weights_mode, prediction_mode=prediction_mode,
                                       save_model_every_batch=save_model_every_batch,
                                       load_weights_before_training=load_weights_before_training,
                                       path_to_weights=output_directory+'last_epoch_model_weights.h5',
                                       validation_value_best_model=best_value)
        results=results.append({'data directory':path_to_data, 'window size':window_size, 'validation_result':val_result}, ignore_index=True)
        results.to_csv('test_results.csv', index=False)

