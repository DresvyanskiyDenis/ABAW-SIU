import os

from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd

from Audio_based.CNN_1D.Classification.Preprocessing.labels_utils import transform_probabilities_to_original_sample_rate
from Audio_based.CNN_1D.utils.utils import CCC_2_sequences_numpy


class Metric_calculator():
    """This class is created to calculate metrics.
       Moreover, it can average cutted predictions with the help of their  cutted_labels_timesteps.
       cutted_labels_timesteps represents timestep of each cutted prediction value. Cutted predictions comes from
       model, which can predict values from data only partual, with defined window size
       e.g. we have
        cutted_prediction=np.array([
            [1, 2, 3, 4, 5],
            [6, 5 ,43, 2, 5],
            [2, 65, 1, 4, 6],
            [12, 5, 6, 34, 23]
        ])
        cutted_labels_timesteps=np.array([
            [0,  0.2, 0.4, 0.6, 0.8],
            [0.2, 0.4, 0.6, 0.8, 1],
            [0.4, 0.6, 0.8, 1, 1.2],
            [0.6, 0.8, 1, 1.2, 1.4],
        ])

    it takes, for example all predictions with timestep 0.2 and then average it -> (2+6)/2=4
    the result array will:
    self.predictions=[ 1.0, 4.0, 3.333, 31.0, 3.25, 5.0, 20.0, 23.0]
    timesteps=       [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]


    """

    def __init__(self, cutted_predictions, cutted_labels_timesteps, ground_truth):
        self.ground_truth=ground_truth
        self.predictions=None
        self.cutted_predictions=cutted_predictions
        self.cutted_labels_timesteps=cutted_labels_timesteps
        self.predictions_probabilities=None

    def average_cutted_predictions_by_timestep(self, mode='regression'):
        """This function averages cut predictions. For more info see description of class
        :return: None
        """
        if mode=='regression':
            cutted_predictions_flatten=self.cutted_predictions.reshape((-1, self.cutted_predictions.shape[-1]))
            cutted_labels_timesteps_flatten=self.cutted_labels_timesteps.reshape((-1,1))
            dataframe_for_avg=pd.DataFrame(data=np.concatenate((cutted_labels_timesteps_flatten, cutted_predictions_flatten), axis=1))
            dataframe_for_avg=dataframe_for_avg.rename(columns={0:'timestep'})
            dataframe_for_avg = dataframe_for_avg.groupby(by=['timestep']).mean()
            dataframe_for_avg=dataframe_for_avg[dataframe_for_avg.index!=-1]
            labels_regression=dataframe_for_avg.iloc[:].values
            self.predictions=labels_regression
        elif 'categorical' in mode:
            cutted_predictions_flatten=self.cutted_predictions.reshape((-1, self.cutted_predictions.shape[-1]))
            cutted_labels_timesteps_flatten=self.cutted_labels_timesteps.reshape((-1,1))
            dataframe_for_avg=pd.DataFrame(data=np.concatenate((cutted_labels_timesteps_flatten, cutted_predictions_flatten), axis=1))
            dataframe_for_avg=dataframe_for_avg.rename(columns={0:'timestep'})
            dataframe_for_avg = dataframe_for_avg.groupby(by=['timestep']).mean()
            predictions_probabilities=dataframe_for_avg.iloc[:].values
            if mode=='categorical_labels':
                predictions_probabilities=np.argmax(predictions_probabilities, axis=-1)
                self.predictions=predictions_probabilities
            elif mode=='categorical_probabilities':
                self.predictions_probabilities=predictions_probabilities


    def calculate_FG_2020_categorical_score_across_all_instances(self, instances, delete_value=-1):
        ground_truth_all=np.zeros((0,))
        predictions_all=np.zeros((0,))
        for instance in instances:
            ground_truth_all=np.concatenate((ground_truth_all, instance.labels))
            predictions_all = np.concatenate((predictions_all, instance.predictions))
        mask=ground_truth_all!=delete_value
        ground_truth_all=ground_truth_all[mask]
        predictions_all=predictions_all[mask]
        return 0.67*f1_score(ground_truth_all, predictions_all, average='macro')+0.33*accuracy_score(ground_truth_all, predictions_all)

    def calculate_FG_2020_F1_and_accuracy_scores_across_all_instances(self, instances, delete_value=-1):
        ground_truth_all=np.zeros((0,))
        predictions_all=np.zeros((0,))
        for instance in instances:
            ground_truth_all=np.concatenate((ground_truth_all, instance.labels))
            predictions_all = np.concatenate((predictions_all, instance.predictions))
        mask=ground_truth_all!=delete_value
        ground_truth_all=ground_truth_all[mask]
        predictions_all=predictions_all[mask]
        return 0.67*f1_score(ground_truth_all, predictions_all, average='macro')+0.33*accuracy_score(ground_truth_all, predictions_all), \
               f1_score(ground_truth_all, predictions_all, average='macro'), \
               accuracy_score(ground_truth_all, predictions_all)

    def calculate_FG_2020_F1_and_accuracy_scores_with_extended_predictions(self, instances, path_to_video, path_to_real_labels, original_sample_rate, delete_value=-1):
        dict_filename_to_predictions = transform_probabilities_to_original_sample_rate(database_instances=instances,
                                                                                       path_to_video=path_to_video,
                                                                                       original_sample_rate=original_sample_rate,
                                                                                       need_save=False)
        real_filenames = os.listdir(path_to_real_labels)
        total_predictions = pd.DataFrame()
        total_labels = pd.DataFrame()
        for real_labels_filename in real_filenames:
            predictions_filename = real_labels_filename.split('.')[0] + '.csv'
            predictions = dict_filename_to_predictions[predictions_filename]
            if total_predictions.shape[0] == 0:
                total_predictions = predictions
            else:
                total_predictions = total_predictions.append(predictions)

            real_labels = pd.read_csv(path_to_real_labels + real_labels_filename, header=None)
            if total_labels.shape[0] == 0:
                total_labels = real_labels
            else:
                total_labels = total_labels.append(real_labels)

        total_predictions = np.argmax(total_predictions.values, axis=-1).reshape((-1, 1))
        mask = total_labels != delete_value
        total_labels = total_labels.values[mask]
        total_predictions = total_predictions[mask]

        return 0.67 * f1_score(total_labels, total_predictions, average='macro') + 0.33 * accuracy_score(total_labels,total_predictions), \
               f1_score(total_labels, total_predictions, average='macro'), \
               accuracy_score(total_labels, total_predictions)

    @staticmethod
    def calculate_FG_2020_CCC_score_with_extended_predictions(instances, path_to_video, path_to_real_labels, original_sample_rate, delete_value=-5):

        dict_filename_to_predictions = transform_probabilities_to_original_sample_rate(database_instances=instances,
                                                                                       path_to_video=path_to_video,
                                                                                       original_sample_rate=original_sample_rate,
                                                                                       need_save=False,
                                                                                       labels_type='regression')
        real_filenames = os.listdir(path_to_real_labels)
        total_predictions = pd.DataFrame()
        total_labels = pd.DataFrame()
        for real_labels_filename in real_filenames:
            predictions_filename = real_labels_filename.split('.')[0] + '.csv'
            predictions = dict_filename_to_predictions[predictions_filename]
            if total_predictions.shape[0] == 0:
                total_predictions = predictions
            else:
                total_predictions = total_predictions.append(predictions)

            real_labels = pd.read_csv(path_to_real_labels + real_labels_filename, header=None)
            if total_labels.shape[0] == 0:
                total_labels = real_labels
            else:
                total_labels = total_labels.append(real_labels)

        mask = (total_labels.values != delete_value).all(axis=-1)
        total_labels = total_labels.values[mask]
        total_predictions = total_predictions.values[mask]

        result=np.zeros(shape=(total_labels.shape[-1],))
        for i in range(total_labels.shape[-1]):
            result[i]=CCC_2_sequences_numpy(total_labels[...,i], total_predictions[...,i])
        return result

    def calculate_accuracy(self):
        return accuracy_score(self.ground_truth, self.predictions)

    def calculate_f1_score(self, mode='macro'):
        return f1_score(self.ground_truth, self.predictions, average=mode)

    def calculate_FG_2020_categorical_score(self, f1_score_mode='macro'):
        return 0.67*self.calculate_f1_score(f1_score_mode)+0.33*self.calculate_accuracy()