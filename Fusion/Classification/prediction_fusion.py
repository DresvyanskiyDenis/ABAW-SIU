import os

import cv2
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from Audio_based_models.CNN_1D.Classification.Preprocessing.labels_utils import construct_video_filename_from_label, \
    get_video_frame_rate, extend_sample_rate_of_labels


def align_sample_rate_to_video_rate(predictions, path_to_video, filename, original_sample_rate):
    video_filename = construct_video_filename_from_label(path_to_video=path_to_video,
                                                         label_filename=filename)
    video_frame_rate = get_video_frame_rate(path_to_video + video_filename)

    predictions = extend_sample_rate_of_labels(predictions, original_sample_rate, video_frame_rate)
    predictions = predictions.astype('float32')

    # align to video amount of frames
    cap = cv2.VideoCapture(path_to_video + video_filename)
    video_frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    aligned_predictions = np.zeros(shape=(video_frame_length, predictions.shape[1]), dtype='float32')
    predictions = predictions.values
    if video_frame_length <= predictions.shape[0]:
        aligned_predictions[:] = predictions[:video_frame_length]
    else:
        aligned_predictions[:predictions.shape[0]] = predictions[:]
        value_to_fill = predictions[-1]
        aligned_predictions[predictions.shape[0]:] = value_to_fill
    aligned_predictions = pd.DataFrame(data=aligned_predictions)
    return aligned_predictions


def main():
    path_to_VGGFace2_based_predictions=''
    path_to_FER_based_predictions=''
    path_to_PANN_based_predictions=''
    path_to_1D_CNN_based_predictions=''
    path_to_ground_truth=''
    val_prefix='val\\'
    test_prefix='test\\'
    path_to_video=''
    path_to_filenames_of_test_data=''
    is_data_test=False
    audio_predictions_sample_rate=5
    type_of_fusion='only_model' # 'class_model' is also possible

    labels_filenames=os.listdir(path_to_1D_CNN_based_predictions+'predictions_val\\')
    if is_data_test:
        path_to_filenames_labels=path_to_filenames_of_test_data
        labels_filenames=pd.read_csv(path_to_filenames_labels, header=None).values.reshape((-1))
    VGGFace2_based_predictions=pd.DataFrame()
    PANN_predictions=pd.DataFrame()
    _1D_CNN_based_predictions=pd.DataFrame()
    FER_based_predictions=pd.DataFrame
    ground_truth=pd.DataFrame()
    for lbs_filename in labels_filenames:
        filename=lbs_filename.split('.')[0]
        gt=pd.read_csv(path_to_ground_truth+'val\\'+filename+'.txt', header=None)

        vgg=pd.read_csv(path_to_VGGFace2_based_predictions+val_prefix+filename+'.txt', header=None)
        vgg=pd.DataFrame(data=vgg.iloc[:,1:].values)

        pann=pd.read_csv(path_to_PANN_based_predictions+val_prefix+filename+'.csv', header=None)
        pann=align_sample_rate_to_video_rate(pann, path_to_video, filename, audio_predictions_sample_rate)

        _1d_cnn=pd.read_csv(path_to_1D_CNN_based_predictions+val_prefix+filename+'.csv', header=None)
        _1d_cnn = align_sample_rate_to_video_rate(_1d_cnn, path_to_video, filename, audio_predictions_sample_rate)

        fer=pd.read_csv(path_to_FER_based_predictions+val_prefix+filename+'.csv', header=None)
        fer=align_sample_rate_to_video_rate(fer, path_to_video, filename, audio_predictions_sample_rate)

        if _1D_CNN_based_predictions.shape[0]==0:
            PANN_predictions=pann
            VGGFace2_based_predictions=vgg
            _1D_CNN_based_predictions=_1d_cnn
            ground_truth=gt
            FER_based_predictions=fer
        else:
            PANN_predictions=PANN_predictions.append(pann)
            VGGFace2_based_predictions=VGGFace2_based_predictions.append(vgg)
            _1D_CNN_based_predictions=_1D_CNN_based_predictions.append(_1d_cnn)
            ground_truth=ground_truth.append(gt)
            FER_based_predictions=FER_based_predictions.append(fer)

    predictions=[VGGFace2_based_predictions, _1D_CNN_based_predictions, PANN_predictions, FER_based_predictions]
    num_predictions=len(predictions)
    num_weights=1000
    num_classes=7
    if type_of_fusion=='only_model':
        weights=np.zeros(shape=(num_weights, num_predictions ))
        for i in range(num_weights):
            weights[i]=np.random.dirichlet(alpha=np.ones((num_predictions,)), size=1)
    elif type_of_fusion=='class_model':
        weights = np.zeros(shape=(num_weights, num_predictions, num_classes))
        for i in range(num_weights):
            weights[i] = np.random.dirichlet(alpha=np.ones((num_predictions,)), size=num_classes)

    best=0
    best_weights=None
    for weight_idx in range(num_weights):
        final_prediction=predictions[0]*weights[weight_idx, 0]
        for i in range(1, num_predictions):
            final_prediction+=predictions[i]*weights[weight_idx, i]
        final_prediction=np.argmax(final_prediction.values, axis=-1)
        delete_mask=ground_truth.values!=-1

        metric=0.67 * f1_score(final_prediction.reshape((-1,1))[delete_mask], ground_truth.values[delete_mask], average='macro') + 0.33 * accuracy_score(final_prediction.reshape((-1,1))[delete_mask],ground_truth.values[delete_mask])
        if metric>best:
            best=metric
            best_weights=weights[weight_idx]
            print('best metric now: %f'%(best))
            print('weights:', best_weights)
    print('final best metric:%f'%(best))
    print('weights:', best_weights)

    # generate test predictions
    path_to_save=''
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    folder_to_save='trial\\'
    if not os.path.exists(path_to_save+folder_to_save):
        os.mkdir(path_to_save+folder_to_save)
    np.savetxt(path_to_save+folder_to_save+'weights_for_fusion.txt', best_weights)
    folder_to_save_predictions='test_predictions\\'
    if not os.path.exists(path_to_save+folder_to_save+folder_to_save_predictions):
        os.mkdir(path_to_save+folder_to_save+folder_to_save_predictions)

    columns_for_test='Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise\n'
    path_to_filenames_test_labels=path_to_filenames_of_test_data
    labels_filenames=pd.read_csv(path_to_filenames_test_labels, header=None).values.reshape((-1))
    for lbs_filename in labels_filenames:
        filename = lbs_filename.split('.')[0]

        vgg=pd.read_csv(path_to_VGGFace2_based_predictions+test_prefix+filename+'.txt')
        vgg=pd.DataFrame(data=vgg.iloc[:,1:].values)

        pann=pd.read_csv(path_to_PANN_based_predictions+test_prefix+filename+'.csv', header=None)
        pann=align_sample_rate_to_video_rate(pann, path_to_video, filename, 5)

        _1d_cnn=pd.read_csv(path_to_1D_CNN_based_predictions+test_prefix+filename+'.csv', header=None)
        _1d_cnn = align_sample_rate_to_video_rate(_1d_cnn, path_to_video, filename, 5)

        fer=pd.read_csv(path_to_FER_based_predictions+test_prefix+filename+'.csv', header=None)
        fer=align_sample_rate_to_video_rate(fer, path_to_video, filename, 5)


        predictions=[vgg, _1d_cnn, pann,fer]

        final_test_prediction=predictions[0]*best_weights[0]
        for i in range(1, num_predictions):
            final_test_prediction += predictions[i] * best_weights[i]
        final_test_prediction = np.argmax(final_test_prediction.values, axis=-1).reshape((-1,1)).astype('int32')
        file=open(path_to_save+folder_to_save+folder_to_save_predictions+filename+'.csv', 'w')
        file.write(columns_for_test)
        file.close()
        pd.DataFrame(data=final_test_prediction).to_csv(path_to_save+folder_to_save+folder_to_save_predictions+filename+'.csv', header=False, index=False, mode='a')





if __name__ == "__main__":
    main()