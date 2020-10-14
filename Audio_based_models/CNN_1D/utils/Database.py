import numpy as np
import os

from sklearn.preprocessing import StandardScaler

from Audio_based_models.CNN_1D.utils.Database_instance import Database_instance
from Audio_based_models.CNN_1D.utils.utils import plot_confusion_matrix


class Database():

    def __init__(self, path_to_data, path_to_labels, data_filetype='wav', data_postfix=''):

        self.path_to_data=path_to_data
        self.path_to_labels=path_to_labels
        self.data_frame_rate=None
        self.labels_frame_rate=None
        self.data_instances=[]
        self.data_filetype=data_filetype
        self.data_postfix=data_postfix

    def load_all_data_and_labels(self, loading_data_function, loading_labels_function):
        """This function loads data and labels from folder self.path_to_data and file with path path_to_labels
           For computational efficiency the loading of labels is made as a separate function load_labels_get_dict()
           Every file is represented as instance of class Database_instance(). The data loading realized by Database_instance() class.
           Since all files have the same frame rates (as well as labels), data_frame_rate and labels_frame_rate will set
           to the same value taken from first element of list data_instances

        :return:None
        """
        # Since all labels are represented by only one file, for computational effeciency firstly we load all labels
        # and then give them to different loaded audiofiles
        list_labels_filenames=os.listdir(self.path_to_labels)
        counter=0
        for labels_filename in list_labels_filenames:
            instance = Database_instance()
            instance.loading_data_function=loading_data_function
            instance.load_data(self.path_to_data + labels_filename.split('_left')[0].split('_right')[0].split('.')[0]+self.data_postfix+'.'+self.data_filetype)
            instance.label_filename=labels_filename.split('.')[0]
            instance.labels, instance.labels_frame_rate=loading_labels_function(self.path_to_labels+labels_filename)
            instance.generate_timesteps_for_labels()
            self.data_instances.append(instance)
            print(counter, len(list_labels_filenames))
            counter+=1
        self.data_frame_rate=self.data_instances[0].data_frame_rate
        self.labels_frame_rate = self.data_instances[0].labels_frame_rate

    def cut_all_instances(self, window_size, window_step):
        """This function is cutting all instances of database (elements of list, which is Database_instance())
        It exploits included in Database_instance() class function for cutting.

        :param window_size: float, size of window in seconds
        :param window_step: float, step of window in seconds
        :return: None
        """
        for i in range(len(self.data_instances)):
            self.data_instances[i].cut_data_and_labels_on_windows(window_size, window_step)

    def normalize_data_within_database(self, scaler=None, return_scaler=False):
        data, _ = self.get_all_concatenated_data_and_labels()
        if scaler==None:
            scaler=StandardScaler()
            scaler=scaler.fit(data)
        for instance in self.data_instances:
            instance.data=scaler.transform(instance.data)
        if return_scaler==True:
            return scaler

    def get_all_concatenated_data_and_labels(self):
        """This function concatenates data and labels of all elements of list self.data_instances
           Every element of list is Database_instance() class, which contains field data and labels

        :return: 2D ndarray, shape=(num_instances_in_list*num_per_instance, data_window_size),
                    concatenated data of every element of list self.data_instances
                 2D ndarray, shape=(num_instances_in_list*num_per_instance, labels_window_size),
                    concatenated labels of every element of list self.data_instances
        """
        tmp_data=[]
        tmp_labels=[]
        for i in range(len(self.data_instances)):
            tmp_data.append(self.data_instances[i].data)
            tmp_labels.append(self.data_instances[i].labels.reshape((-1,1)))
        result_data=np.vstack(tmp_data)
        result_labels = np.vstack(tmp_labels).reshape((-1))
        return result_data, result_labels

    def reduce_labels_frame_rate(self, needed_frame_rate):
        """This function reduce labels frame rate to needed frame rate by taking every (second, thirs and so on) elements from
           based on calculated ratio.
           ratio calculates between current frame rate and needed frame rate

        :param needed_frame_rate: int, needed frame rate of labels per one second (e.g. 25 labels per second)
        :return:None
        """
        ratio=int(self.labels_frame_rate/needed_frame_rate)
        self.labels_frame_rate=needed_frame_rate
        for i in range(len(self.data_instances)):
            self.data_instances[i].labels=self.data_instances[i].labels[::ratio]
            self.data_instances[i].labels_frame_rate=needed_frame_rate


    def prepare_data_for_training(self, window_size, window_step, delete_value=-1, need_scaling=False, scaler=None, return_scaler=False):
        # check if every instance has data_frame_rate or not
        # if not, it means that frame rate of data equels to labels frame rate

        if self.data_frame_rate==None:
            self.data_frame_rate = self.labels_frame_rate
            for instance in self.data_instances:
                instance.data_frame_rate=self.data_frame_rate

        # aligning labels
        for instance in self.data_instances:
            instance.align_number_of_labels_and_data()
            instance.generate_timesteps_for_labels()
        # delete all -1 labels
        if delete_value!=None:
            for instance in self.data_instances:
                instance.data=instance.generate_array_without_class(instance.data,instance.data_frame_rate, delete_value)
                instance.labels = instance.generate_array_without_class(instance.labels,instance.labels_frame_rate, delete_value)
                # check equallity of length of data and labels (It can arrise due to inaccuracy of converting )
                instance.check_equallity_data_length_and_labels()
                instance.generate_timesteps_for_labels()

        # check if some file have 0 labels (this could be, if all labels were -1. You can face it in FG_2020 competition)
        tmp_list=[]
        for instance in self.data_instances:
            if instance.labels.shape[0]!=0:
                tmp_list.append(instance)
        self.data_instances=tmp_list

        # scaling
        if need_scaling==True:
            scaler=self.normalize_data_within_database(scaler=scaler, return_scaler=return_scaler)


        # cutting
        self.cut_all_instances(window_size, window_step)
        # return scaler if need
        if return_scaler==True:
            return scaler

    def plot_confusion_matrix(self, delete_value=-1):
        ground_truth_all = np.zeros((0,))
        predictions_all = np.zeros((0,))
        for instance in self.data_instances:
            ground_truth_all = np.concatenate((ground_truth_all, instance.labels))
            predictions_all = np.concatenate((predictions_all, instance.predictions))
        mask = ground_truth_all != delete_value
        ground_truth_all = ground_truth_all[mask]
        predictions_all = predictions_all[mask]

        ax = plot_confusion_matrix(y_true=ground_truth_all,
                                   y_pred=predictions_all,
                                   classes=np.unique(ground_truth_all))