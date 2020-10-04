import numpy as np
import pandas as pd

from Audio_based.CNN_1D.utils.Metric_calculator import Metric_calculator


def extract_cutted_values_from_given_indexes(dataframe_indexes, dict_instances, result_array_shape, type_data='data'):
    result_array=np.zeros(result_array_shape)
    for i in range(dataframe_indexes.shape[0]):
        start_idx, end_idx=dataframe_indexes.iloc[i,:].values
        filename=dataframe_indexes.iloc[i].index.values
        if type_data=='data':
            result_array[i]=dict_instances[filename].data[start_idx:end_idx]
        elif type_data=='labels':
            result_array[i] = dict_instances[filename].labels[start_idx:end_idx]
    return result_array

def extract_cutted_data_labels_from_given_indexes(dataframe_indexes, dict_instances, result_data_shape, result_lbs_shape):
    """This function extracts data and labels in window format via corresponding indexes located in dataframe_indexes
       Cut format is needed for train recurrent models. The technique of converting:
       for example, if we have sequence [1 2 3 4 5 6 7 8], window_size=4, window_step=3 then
       dataframe_indexes can be as [0, 4] - it is indexes
                                   [3, 7]
                                   [4, 8]

        1st window: |1 2 3 4| 5 6 7 8
                  ......
        2nd window: 1 2 3 |4 5 6 7| 8
                        ..
        3rd window: 1 2 3 4 |5 6 7 8|

    :param dataframe_indexes: DataFrame, contains filename and indexes of data and labesl windows
    :param dict_instances: dictionary, key: filename, value:Database_instance()
    :param result_data_shape: tuple, the shape of generated labels
    :param result_lbs_shape: tuple, the shape of generated labels
    :return: result_data: ndarray, windowed data generated from indexes dataframe_indexes
             result_labels: ndarray, windowed labels generated from indexes dataframe_indexes
    """
    result_data = np.zeros(result_data_shape, dtype='float32')
    result_labels = np.zeros(result_lbs_shape, dtype='float32')
    for i in range(dataframe_indexes.shape[0]):
        start_idx_data, end_idx_data,start_idx_lbs, end_idx_lbs = dataframe_indexes.iloc[i, :].values
        filename = dataframe_indexes.index[i]
        # Padding if the shape of data less then window size
        if start_idx_data==0 and end_idx_data>dict_instances[filename].data.shape[0]:
            result_data[i]=np.zeros(shape=(end_idx_data-start_idx_data,)+dict_instances[filename].data.shape[1:])
            result_data[i,:dict_instances[filename].data.shape[0]]=dict_instances[filename].data
            result_labels[i]=np.zeros(shape=(end_idx_lbs-start_idx_lbs,)+dict_instances[filename].labels.shape[1:])
            result_labels[i,:dict_instances[filename].labels.shape[0]]=dict_instances[filename].labels
            return result_data, result_labels
        result_data[i] = dict_instances[filename].data[start_idx_data:end_idx_data]
        result_labels[i] = dict_instances[filename].labels[start_idx_lbs:end_idx_lbs]

    return result_data, result_labels

def extract_cutted_data_and_timesteps_from_given_indexes(dataframe_indexes, dict_instances, result_data_shape, result_timesteps_shape):
    """This function extracts data and labels in window format via corresponding indexes located in dataframe_indexes
       Cut format is needed for train recurrent models. The technique of converting:
       for example, if we have sequence [1 2 3 4 5 6 7 8], window_size=4, window_step=3 then
       dataframe_indexes can be as [0, 4] - it is indexes
                                   [3, 7]
                                   [4, 8]

        1st window: |1 2 3 4| 5 6 7 8
                  ......
        2nd window: 1 2 3 |4 5 6 7| 8
                        ..
        3rd window: 1 2 3 4 |5 6 7 8|

    :param dataframe_indexes: DataFrame, contains filename and indexes of data and labesl windows
    :param dict_instances: dictionary, key: filename, value:Database_instance()
    :param result_data_shape: tuple, the shape of generated labels
    :param result_lbs_shape: tuple, the shape of generated labels
    :return: result_data: ndarray, windowed data generated from indexes dataframe_indexes
             result_labels: ndarray, windowed labels generated from indexes dataframe_indexes
    """
    result_data = np.zeros(result_data_shape, dtype='float32')
    result_timesteps = np.zeros(result_timesteps_shape, dtype='float32')
    for i in range(dataframe_indexes.shape[0]):
        start_idx_data, end_idx_data,start_idx_lbs, end_idx_lbs = dataframe_indexes.iloc[i, :].values
        filename = dataframe_indexes.index[i]
        # Padding if the shape of data less then window size
        if start_idx_data==0 and end_idx_data>dict_instances[filename].data.shape[0]:
            result_data[i]=np.zeros(shape=(end_idx_data-start_idx_data,)+dict_instances[filename].data.shape[1:])
            result_data[i,:dict_instances[filename].data.shape[0]]=dict_instances[filename].data
            result_timesteps[i]=np.zeros(shape=(1,)+(end_idx_lbs-start_idx_lbs,))-1
            result_timesteps[i,:dict_instances[filename].labels.shape[0]]=dict_instances[filename].labels_timesteps
            return result_data, result_timesteps
        result_data[i] = dict_instances[filename].data[start_idx_data:end_idx_data]
        result_timesteps[i] = dict_instances[filename].labels_timesteps[start_idx_lbs:end_idx_lbs]

    return result_data, result_timesteps


def batch_generator_cut_data(instances, batch_size=32, need_shuffle=False, need_sample_weight=False, class_weights=None):
    """This generator extracts indexes of windows from all instances, which are Database_instance() objects
       shuffle it, if it is needed, and piece by piece generate cut data and labels from extracted indexes
       and yield it to save RAM
       Cut format is needed for train recurrent models. The technique of converting:
       for example, if we have sequence [1 2 3 4 5 6 7 8], window_size=4, window_step=3 then
       dataframe_indexes can be as [0, 4] - it is indexes
                                   [3, 7]
                                   [4, 8]

        1st window: |1 2 3 4| 5 6 7 8
                  ......
        2nd window: 1 2 3 |4 5 6 7| 8
                        ..
        3rd window: 1 2 3 4 |5 6 7 8|

    :param instances: list of Database_instance(), instances of database
    :param batch_size:int
    :param need_shuffle:boolean
    :return: cut_data: ndarray, data in window format
             cut_labels: ndarray, labels in window format
    """
    # indexes, which contains window indexes of data and labels (as well as filename of file)
    windows_indexes=pd.DataFrame(columns=['filename', 'data_start_idx', 'data_end_idx', 'label_start_idx', 'label_end_idx'])
    windows_indexes=windows_indexes.set_index('filename')
    data_window_size=instances[0].data_window_size
    labels_window_size=instances[0].labels_window_size
    # concatenate indexes of windows from all files
    for instance in instances:
        data_indexes=instance.cutted_data_indexes
        labels_indexes=instance.cutted_labels_indexes
        concatenated=np.concatenate((data_indexes,labels_indexes), axis=1)
        windows_indexes=windows_indexes.append(pd.DataFrame(columns=windows_indexes.columns, data=concatenated, index=[instance.filename for i in range(concatenated.shape[0])]))

    if need_shuffle==True:
        windows_indexes = windows_indexes.sample(frac=1)

    # for access to instances via filename
    filename_instance_dict = {}
    for instance in instances:
        filename_instance_dict[instance.filename] = instance
    # go through indexes to generate cutted train data and labels with corresponding batch_size
    for i in range(0, windows_indexes.shape[0], batch_size):
        # batch_size can be more then amount of instances on last step
        start_idx_windows_indexes=i
        if start_idx_windows_indexes+batch_size>windows_indexes.shape[0]:
            end_idx_windows_indexes=windows_indexes.shape[0]
        else:
            end_idx_windows_indexes=start_idx_windows_indexes+batch_size
        # shapes of needed arrays of cutted data and labels
        future_cut_data_shape=(end_idx_windows_indexes-start_idx_windows_indexes, data_window_size,instances[0].data.shape[-1])
        future_cut_lbs_shape=(end_idx_windows_indexes-start_idx_windows_indexes, labels_window_size,instances[0].labels.shape[-1])

        # start of extracting cutted data and labels from generated indexes
        current_indexes_of_windows=windows_indexes.iloc[i:(i+batch_size),:]
        cutted_data, cutted_labels= extract_cutted_data_labels_from_given_indexes(dataframe_indexes=current_indexes_of_windows,
                                                                                  dict_instances=filename_instance_dict,
                                                                                  result_data_shape=future_cut_data_shape,
                                                                                  result_lbs_shape=future_cut_lbs_shape)
        # if we need sample weights for balancing data
        if need_sample_weight==True:
            sample_weight = cutted_labels.copy().astype('float32')
            for i in range(class_weights.shape[0]):
                mask = (sample_weight == i)
                sample_weight[mask] = class_weights[i]
            yield cutted_data, cutted_labels, sample_weight
        else:
            yield cutted_data, cutted_labels

def predict_data_with_the_model(model, instances, prediction_mode='sequence_to_sequence', labels_type='categorical'):
    data_window_size = instances[0].data_window_size
    labels_window_size=instances[0].labels_window_size
    # for access to instances via filename
    filename_instance_dict = {}
    for instance in instances:
        filename_instance_dict[instance.filename] = instance

    for instance in instances:
        dataframe_indexes=pd.DataFrame(columns=['filename', 'data_start_idx', 'data_end_idx', 'label_start_idx', 'label_end_idx'])
        dataframe_indexes=dataframe_indexes.set_index('filename')
        data_indexes = instance.cutted_data_indexes
        labels_indexes = instance.cutted_labels_indexes
        concatenated = np.concatenate((data_indexes, labels_indexes), axis=1)
        dataframe_indexes = dataframe_indexes.append(pd.DataFrame(columns=dataframe_indexes.columns, data=concatenated, index=[instance.filename for i in range(concatenated.shape[0])]))
        # shapes of needed arrays of cutted data and labels
        future_cut_data_shape = (dataframe_indexes.shape[0], data_window_size, 1)
        future_cut_lbs_shape = (dataframe_indexes.shape[0], labels_window_size)

        # start of extracting cutted data and labels from generated indexes
        cut_data, cut_timesteps = extract_cutted_data_and_timesteps_from_given_indexes(
            dataframe_indexes=dataframe_indexes,
            dict_instances=filename_instance_dict,
            result_data_shape=future_cut_data_shape,
            result_timesteps_shape=future_cut_lbs_shape)
        predictions=model.predict(cut_data, batch_size=1)

        # if we have only one predicted labels per whole window, we need to extend it
        if prediction_mode=='sequence_to_one':
            extended_predictions=np.zeros(cut_timesteps.shape+(predictions.shape[-1],))
            for i in range(predictions.shape[0]):
                extended_predictions[i,:]=predictions[i]
            predictions=extended_predictions

        cut_timesteps=cut_timesteps.reshape((-1,1))
        metric_calculator = Metric_calculator(predictions, cut_timesteps,
                                              ground_truth=instance.labels)
        if labels_type=='categorical':
            metric_calculator.average_cutted_predictions_by_timestep(mode='categorical_labels')
            metric_calculator.average_cutted_predictions_by_timestep(mode='categorical_probabilities')
            instance.predictions = metric_calculator.predictions
            instance.predictions_probabilities=metric_calculator.predictions_probabilities
        elif labels_type=='regression':
            metric_calculator.average_cutted_predictions_by_timestep(mode='regression')
            instance.predictions = metric_calculator.predictions
