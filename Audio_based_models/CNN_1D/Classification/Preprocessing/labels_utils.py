import math
import numpy as np
import pandas as pd
import os
import cv2


def transform_probabilities_to_original_sample_rate(database_instances, path_to_video, original_sample_rate, need_save=True, path_to_output='', labels_type='categorical'):
    dict_filename_to_aligned_predictions={}
    for instance in database_instances:
        # extending
        if labels_type=='categorical':
            predictions=instance.predictions_probabilities
        elif labels_type=='regression':
            predictions=instance.predictions
        lbs_filename=instance.label_filename
        predictions=pd.DataFrame(data=predictions)
        video_filename = construct_video_filename_from_label(path_to_video=path_to_video,
                                                             label_filename=lbs_filename)
        video_frame_rate = get_video_frame_rate(path_to_video + video_filename)

        predictions = extend_sample_rate_of_labels(predictions, original_sample_rate, video_frame_rate)
        predictions = predictions.astype('float32')
        # align to video amount of frames
        cap = cv2.VideoCapture(path_to_video+ video_filename)
        video_frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        aligned_predictions = np.zeros(shape=(video_frame_length, predictions.shape[1]), dtype='float32')
        if video_frame_length <= predictions.shape[0]:
            aligned_predictions[:] = predictions[:video_frame_length]
        else:
            aligned_predictions[:predictions.shape[0]] = predictions[:]
            value_to_fill = predictions[-1]
            aligned_predictions[predictions.shape[0]:] = value_to_fill
        aligned_predictions = pd.DataFrame(data=aligned_predictions)
        if need_save:
            if not os.path.exists(path_to_output):
                os.mkdir(path_to_output)
            #f = open(path_to_output + lbs_filename.split('_vocal')[0]+'.csv', 'w')
            #f.write('Sample rate:%i' % video_frame_rate + '\n')
            #f.close()
            aligned_predictions.to_csv(path_to_output+lbs_filename.split('_vocal')[0]+'.csv', header=False, index=False)
            # you need to return also
        dict_filename_to_aligned_predictions[lbs_filename+'.csv']=aligned_predictions
    return dict_filename_to_aligned_predictions


def fill_gap_between_two_points(left_value, right_value, num_points):
    """This function transform two points to sequence with filling gap between it
    e.g.
        left_value=1
        right_value=1
        num_points=10
        1 _ _ _ _ _ _ _ _ 1 ----> 1 1 1 1 1 1 1 1 1 1
    if we have different values, then
        left_value=1
        right_value=2
        num_points=10
        1 _ _ _ _ _ _ _ _ 2 ----> 1 1 1 1 1 2 2 2 2 2


    :param left_value: int or float
    :param right_value: int or float
    :param num_points: int
    :return: ndarray, sequence with left and right values on borders and filled gap between them
    """
    if left_value==right_value:
        tmp=[left_value for i in range(num_points)]
        res=np.array(tmp)
    else:
        middle=num_points//2
        res=np.ones((num_points,))
        res[0:middle]=left_value
        res[middle:]=right_value
    return res

def extend_sample_rate_of_labels(labels, original_sample_rate, needed_sample_rate):
    """This function extends sample rate of provided labels from original_sample_rate to needed_sample_rate
       by stretching existing labels with calculated ratio

    :param labels: ndarray (n_labels,) or DataFrame (n_labels, 1)
    :param original_sample_rate: int, sample rate of provided labels
    :param needed_sample_rate:int
    :return: ndarray, stretched labels with new sample_rate
    """
    # calculate ration between original and needed sample rates
    ratio=needed_sample_rate/original_sample_rate
    new_size_of_labels=int(math.ceil(labels.shape[0]*ratio))
    # calculating key_points - positions, which will be used as indexes for provided labels_filenames
    # e.g.
    # we have labels [1 1 1 2 2 1] with sample rate=2 and we need sample rate 6 (ratio=3)
    # then key_points will be
    # [0 3 6 9 12 15] ---> [1 _ _ 1 _ _ 1 _ _ 2 _ _ 2 _ _ 1 _ _]
    # old shape=6 ---> new shape = 18
    expanded_numpy_with_nan=generate_extended_array_with_key_points(array_to_extend=labels.values, new_size_of_array=new_size_of_labels, ratio=ratio)
    new_labels=pd.DataFrame(columns=labels.columns,data=expanded_numpy_with_nan).interpolate()
    return new_labels

def generate_extended_array_with_key_points(array_to_extend, new_size_of_array, ratio):
    expanded_numpy = np.full(shape=(new_size_of_array, array_to_extend.shape[1]), fill_value=np.nan)
    int_part=ratio//1
    float_part=ratio%1
    idx_expanded_array=0
    idx_array_to_extend=0
    residual=0
    while idx_array_to_extend<array_to_extend.shape[0]:
        expanded_numpy[idx_expanded_array]=array_to_extend[idx_array_to_extend]
        residual+=float_part
        idx_to_add=int(int_part)
        if residual>=1:
            idx_to_add+=1
            residual-=1
        idx_expanded_array+=idx_to_add
        idx_array_to_extend+=1
    return expanded_numpy


def downgrade_sample_rate_of_labels(labels, original_sample_rate, needed_sample_rate):
    """This function downgrades sample rate of provided labels to needed_sample_rate
       It is worked through calculating ratio between original_sample_rate and needed_sample_rate
       and then thinning provided labels

    :param labels: ndarray, (n_labels,)
    :param original_sample_rate: int
    :param needed_sample_rate: int
    :return: ndarray, thinned labels with new downgraded sample_rate
    """
    ratio=int(original_sample_rate/needed_sample_rate)
    new_labels=labels[::ratio]
    return new_labels

def downgrade_sample_rate_of_all_labels(path_to_labels, path_to_output, needed_sample_rate):
    """This function exploits downgrade_sample_rate_of_labels() function to process all labels in provided directory
       path_to_labels. Processed labels will save in path_to_output directory

    :param path_to_labels: string, path to directory
    :param path_to_output: string, path to directory
    :param needed_sample_rate: int, sample rate for downgrading
    :return: None
    """
    labels_filenames=os.listdir(path_to_labels)
    if not os.path.exists(path_to_output):
        os.mkdir(path_to_output)
    for lbs_filename in labels_filenames:
        labels=pd.read_csv(path_to_labels+lbs_filename, skiprows=1,header=None)
        f = open(path_to_labels+lbs_filename, 'r')
        original_sample_rate=int(f.readline().split(':')[-1])
        f.close()
        labels=downgrade_sample_rate_of_labels(labels=labels,
                                            original_sample_rate=original_sample_rate,
                                            needed_sample_rate=needed_sample_rate)
        f = open(path_to_output+lbs_filename, 'w')
        f.write('Sample rate:%i'%needed_sample_rate+'\n')
        f.close()
        labels.to_csv(path_to_output+lbs_filename, index=False, header=False, mode='a')

def get_video_frame_rate(path_to_video):
    """The function reads params of video to get video frame rate

    :param path_to_video: string, path to certain video
    :return: int, video frame rate
    """
    cap = cv2.VideoCapture(path_to_video)
    video_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    return video_frame_rate

def construct_video_filename_from_label(path_to_video, label_filename):
    """This function generate video filename from label filename
       It is inner function for processing specific data from AffWild2 challenge
       It is needed, because video can be in either mp4 or avi format

    :param path_to_video: string, path to directory with videos
    :param label_filename: string, filename of labels
    :return: string, video filename (e. g. 405.mp4)
    """
    video_filename = label_filename.split('_left')[0].split('_right')[0].split('.')[0]
    if os.path.exists(path_to_video + video_filename + '.mp4'):
        video_filename += '.mp4'
    if os.path.exists(path_to_video + video_filename + '.avi'):
        video_filename += '.avi'
    return video_filename

def extend_sample_rate_of_all_labels(path_to_labels, path_to_video, path_to_output, needed_sample_rate):
    """This function exploits extend_sample_rate_of_labels() function to process (and extend) all labels
       in path_to_labels directory. For processing you also need to specify path_to_video directory, where
       all videos are saved. It is for calculating video_frame_rate.
       All processed labels will be saved in directory path_to_output.

    :param path_to_labels: string, path to directory with labels
    :param path_to_video: string, path to directory with videos (which are corresponded to labels)
    :param path_to_output:string, path to directorz for saving processed labels
    :param needed_sample_rate:int
    :return:None
    """
    labels_filenames=os.listdir(path_to_labels)
    if not os.path.exists(path_to_output):
        os.mkdir(path_to_output)
    for lbs_filename in labels_filenames:
        labels=pd.read_csv(path_to_labels+lbs_filename, skiprows=1,header=None)
        video_filename=construct_video_filename_from_label(path_to_video=path_to_video,
                                                           label_filename=lbs_filename)

        video_frame_rate=get_video_frame_rate(path_to_video+video_filename)
        needed_sample_rate=needed_sample_rate
        labels=extend_sample_rate_of_labels(labels, video_frame_rate, needed_sample_rate)
        labels=labels.astype('int32')
        labels=pd.DataFrame(labels)
        f = open(path_to_output+lbs_filename, 'w')
        f.write('Sample rate:%i'%needed_sample_rate+'\n')
        f.close()
        labels.to_csv(path_to_output+lbs_filename, index=False, header=False, mode='a')

def align_number_videoframes_and_labels(path_to_video, path_to_label):
    """This function align the number of labels to number of videoframes in corresponding video file
       THis is needed, because some videos in FG-2020 competition have more or less (usually on 1) videoframe than number
       of labels.


    :param path_to_video:string, path to directory with videos (which are corresponded to labels)
    :param path_to_label:string, path to directory with labels
    :return:ndarray, aligned labels
    """
    cap = cv2.VideoCapture(path_to_video)
    video_frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    labels=pd.read_csv(path_to_label, skiprows=1,header=None)
    labels=labels.values.reshape((-1,)).astype('int32')
    aligned_labels = np.zeros(shape=(video_frame_length), dtype='int32')
    if video_frame_length<=labels.shape[0]:
        aligned_labels[:]=labels[:video_frame_length]
    else:
        aligned_labels[:labels.shape[0]]=labels[:]
        value_to_fill = labels[-1]
        aligned_labels[labels.shape[0]:]=value_to_fill
    return aligned_labels


def align_number_videoframes_and_labels_all_data(path_to_video, path_to_labels, output_path):
    """This function exploits the align_number_videoframes_and_labels() function to align all labels located in
       the directory path_to_labels. Processed labels will be saved in output_path directory.

    :param path_to_video:string, path to directory with videos (which are corresponded to labels)
    :param path_to_labels:string, path to directory with labels
    :param output_path:string, directory to save processed labels
    :return:None
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    labels_filenames=os.listdir(path_to_labels)
    for lbs_filename in labels_filenames:
        # get a video filename to calculate then frames
        video_filename=lbs_filename.split('_left')[0].split('_right')[0].split('.')[0]
        if os.path.exists(path_to_video+video_filename+'.mp4'):
            video_filename+='.mp4'
        if os.path.exists(path_to_video+video_filename+'.avi'):
            video_filename+='.avi'
        aligned_labels=align_number_videoframes_and_labels(path_to_video=path_to_video+video_filename,
                                                           path_to_label=path_to_labels+lbs_filename)
        aligned_labels=pd.DataFrame(aligned_labels)
        aligned_labels.to_csv(output_path+lbs_filename, index=False, header=False)





if __name__ == "__main__":
    path_to_video='D:\\Databases\\AffWild2\\Videos\\'
    path_to_original_labels='D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\train\\Training_Set\\'
    aligned_labels_outputh_path='D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\train\\Aligned_labels\\'
    extended_aligned_labels_outputh_path='D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\train\\Aligned_labels_extended\\'
    downgraded_aligned_labels_outputh_path='D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\train\\Aligned_labels_reduced\\'
    sample_rate_for_extension=3000
    sample_rate_for_downgrade=5
    align_number_videoframes_and_labels_all_data(path_to_video=path_to_video,
                                                 path_to_labels=path_to_original_labels,
                                                 output_path=aligned_labels_outputh_path)

    extend_sample_rate_of_all_labels(path_to_labels=aligned_labels_outputh_path,
                                     path_to_video=path_to_video,
                                     path_to_output=extended_aligned_labels_outputh_path,
                                     needed_sample_rate=sample_rate_for_extension)

    downgrade_sample_rate_of_all_labels(path_to_labels=extended_aligned_labels_outputh_path,
                                        path_to_output=downgraded_aligned_labels_outputh_path,
                                        needed_sample_rate=sample_rate_for_downgrade)
