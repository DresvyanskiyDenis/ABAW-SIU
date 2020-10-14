import os
import ast
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score, confusion_matrix
from tensorboardX import SummaryWriter
#from tqdm.notebook import tqdm
from tqdm import tqdm

from data.data_sample import DataSample

def define_seed(seed=12):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def read_samples(data_root, labels_path, logging=False):
    datasets = ['train', 'valid', 'test']
    all_labels = pd.read_csv(os.path.join(data_root, labels_path))
    
    all_samples = {}
    for dataset in datasets:        
        labels_df = all_labels[all_labels['file_name'].str.startswith(dataset)].values
        samples = []
        for file_info in labels_df:
            fp = os.path.join(data_root, file_info[3])
            sample = DataSample(wav_path=fp,
                                file_name=file_info[0],
                                file_size=os.path.getsize(fp),
                                label=file_info[2])
    
            samples.append(sample)
        
        all_samples[dataset] = samples
    
    if logging:
        print('Count of samples: {0}'.format(sum([len(v) for k, v in all_samples.items()])))
        print(all_samples['train'][0])
    
    return all_samples

def find_max_shape(*args):
    return tuple(map(max, zip(*args)))

def str_to_list(col):
    res = [np.asarray(ast.literal_eval(i)) for i in col]
    return res, (res[0].shape[0], max([i.shape[1] for i in res]))

def extend_array(arr, feat_shape):
    return np.asarray([np.pad(i, [(0, 0), (0, feat_shape[1] - i.shape[1])], mode='mean') for i in arr])

def display_transforms(loader, batch_idx, std, mean, idx_to_classes, predicts=None):
    assert batch_idx < len(loader) and batch_idx >= 0
    batch_size = loader.batch_size
    preds = None
    inputs = None
    labels = None
    paths = None
    
    loader_iter = iter(loader)
    for i in range(batch_idx + 1):
        if predicts is not None:
            inputs, labels, paths = next(loader_iter)
            preds = predicts[i * batch_size:(i + 1) * batch_size]
        else:
            inputs, preds = next(loader_iter)
    
        fig, axes = plt.subplots(ncols=4, nrows=4, constrained_layout=False, figsize=(15, 15))
        axes = axes.flatten()
        for img, label, ax in zip(inputs, preds, axes):
            image = img.permute(1, 2, 0).numpy()
            image = std * image + mean
            ax.autoscale(enable=True)
            ax.imshow(image.clip(0, 1))
            if predicts is not None:
                ax.title.set_text(str(label))
            elif idx_to_classes is not None:
                ax.title.set_text(idx_to_classes[np.asscalar(label.numpy())])
                
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', color_map=plt.cm.Blues, fig_path=None):
        """
        This function prints and plots the confusion matrix
        """
        if not title:
            title = 'Confusion matrix'

            # Compute confusion matrix
        # Only use the labels that appear in the data
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Confusion matrix with normalization')
        np.set_printoptions(precision=3)
        print(norm_cm)
        print('Confusion matrix without normalization')
        print(cm)
        np.set_printoptions(precision=6)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=color_map)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

#         ax.set_xticks(np.arange(cm.shape[1] + 1)-.5)
#         ax.set_yticks(np.arange(cm.shape[0] + 1)-.5)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, '{0:d}\n{1:.2f}%'.format(cm[i, j], norm_cm[i, j] * 100),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        if fig_path:
            plt.savefig(fig_path, dpi = 300)
        else:
            plt.show(block=False)
            
def get_best_model(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)

def get_model_by_epoch(path, epoch):
    files = os.listdir(path)
    paths = [basename for basename in files if basename.startswith('epoch_{0}'.format(epoch)) and basename.endswith('.pth')]
    return os.path.join(path, paths[0])
            
def train_model(model, loss, optimizer, scheduler, num_epochs, 
                device, train_dataloader, valid_dataloader, 
                class_names, log_root, tb_log_root, metrics,
                features_name, experiment_name, log_iter=None):    
    main_metric_name = metrics[0].__name__
    max_performance = {
        main_metric_name: .0,
    }
    
    max_epoch = .0
    exp_folder = '{0}_{1}'.format(features_name, experiment_name)
    
    os.makedirs(os.path.join(log_root, exp_folder), exist_ok=True)
    save_path = os.path.join(log_root, exp_folder)

    tb_save_path = os.path.join(tb_log_root, exp_folder)
    
    summary = {
        'train': SummaryWriter('{0}_train'.format(tb_save_path)),
        'valid': SummaryWriter('{0}_valid'.format(tb_save_path))
    }
    
    for epoch in range(num_epochs):        
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            targets = []
            predicts = []
            
            if phase == 'train':
                dataloader = train_dataloader
                model.train()  # Set model to training mode
            else:
                dataloader = valid_dataloader
                model.eval()   # Set model to evaluate mode

            running_loss = 0.
            epoch_score = 0.

            # Iterate over data.
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # forward and backward
                preds = None
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    loss_value = loss(preds, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss_value.item()
                
                targets.extend(labels.cpu().numpy())
                predicts.extend(preds.cpu().detach().numpy())
                

            epoch_loss = running_loss / len(dataloader)
            
            targets, predicts = np.asarray(targets), np.argmax(np.asarray(predicts), axis=1)
            
            performance = {}
            for metric in metrics:
                if metric.__name__ == 'accuracy_score':
                    performance[metric.__name__] = metric(targets, predicts)
                else:
                    performance[metric.__name__] = metric(targets, predicts, average='macro')
            
            epoch_score = performance[main_metric_name]
            
            for metric in performance:
                summary[phase].add_scalar(metric, performance[metric], global_step=epoch)
            
            summary[phase].add_scalar('loss', epoch_loss, global_step=epoch)
            
            print('{} Loss: {:.4f}, Performance:'.format(phase, epoch_loss), flush=True)
            print([metric for metric in performance])
            print([performance[metric] for metric in performance])
            
            if (((phase == 'valid') and (epoch_score > max_performance[main_metric_name])) or 
                ((phase == 'valid') and (log_iter is not None) and (epoch in log_iter))):
                
                if epoch_score > max_performance[main_metric_name]:
                    max_performance = performance
                    max_epoch = epoch
                
                cm = confusion_matrix(targets, predicts, [i for i in range(len(class_names))])
                
                res_name = 'epoch_{0}_{1}'.format(epoch, epoch_score)

                plot_confusion_matrix(cm=cm, 
                                      classes=class_names,
                                      title='{}.png'.format(res_name),
                                      fig_path=os.path.join(save_path, '{}.png'.format(res_name)))                
                model.cpu()
                torch.save({
                    'epoch': epoch,
                    'model': model,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler,
                    'loss_function': loss},
                    os.path.join(save_path, '{}.pth'.format(res_name)))
                
                model.to(device)
                
    print('Epoch: {},\nMax performance:'.format(max_epoch))
    print([metric for metric in max_performance])
    print([max_performance[metric] for metric in max_performance])
    
    for phase in ['train', 'valid']:
        summary[phase].close()
    
    return model, max_epoch, max_performance
