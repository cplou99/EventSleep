"""
Created on Mar 2023
@author: 
@project: EventSleep
"""

import pandas as pd
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sn
from tabulate import tabulate
from pathlib import Path
from scipy import stats
import os

def LabelsNames():
    labels_dict = {0: 'HeadMove', 1: 'Hands2Face/Head', 2: 'RollLeft', 3: 'RollRight', 4: 'LegsShake', 5: 'ArmsShake',
        6: 'LieLeft', 7: 'LieRight', 8: 'LieUp', 9: 'LieDown'}
    return labels_dict


def DataAugmentationLabels():
    map_augmented_labels = {2: 3, 3:2, 6:7, 7:6}
    return map_augmented_labels


def CropBed(frames, data, subject):
    if subject in [1, 2, 3, 4]:
        Day = 1
    else: Day = 2
    crop_bed_dict = {
        'Camera': ['Events', 'Events', 'Infrared', 'Infrared'],
        'Day': [1, 2, 1, 2],
        'w1': [95, 95, 127, 108],
        'w2': [595, 595, 558, 538],
        'h1': [60, 80, 14, 37],
        'h2': [420, 440, 329, 352]
    }

    crop_bed_df = pd.DataFrame(crop_bed_dict)
    crop_bed_day = crop_bed_df.loc[(crop_bed_df['Camera'] == data) & (crop_bed_df['Day'] == Day)]
    w1, w2 = crop_bed_day['w1'].values[0], crop_bed_day['w2'].values[0]
    h1, h2 = crop_bed_day['h1'].values[0], crop_bed_day['h2'].values[0]
    if data == "Events":
        if len(frames.shape) == 3:
            frames = frames[h1:h2, w1:w2, :]
        elif len(frames.shape) == 4:
            frames = frames[h1:h2, w1:w2, :, :]
        elif len(frames.shape) == 5:
            frames = frames[:, h1:h2, w1:w2, :, :]
    elif data == "Infrared":
        if len(frames.shape) == 3:
            frames = frames[h1:h2, w1:w2, :]
        if len(frames.shape) == 4:
            frames = frames[:, h1:h2, w1:w2, :]
    return frames


def TrainOrTest(subject):
    if subject in [9, 12, 13, 14, 15]:
        return "TEST"
    else:
        return "TRAIN"


def GiveWeightsToLabels():
    frequency_labels = torch.tensor([0.0759, 0.1512, 0.1576, 0.0749, 0.0820, 0.0868, 0.0920, 0.0503, 0.1845, 0.0448])
    weights_labels = torch.max(frequency_labels) / frequency_labels
    return weights_labels


def GetLabelFromDirName(dir, len, data, data_augmentation):
    map_augmented_labels = DataAugmentationLabels()
    if data == "Events":
        label = int(dir.split(".")[0][-1])
    elif data == "Infrared":
        label = int(dir[-1])
    if data_augmentation and label in [2, 3, 6, 7]:
        y_dir1 = np.repeat(label, len/2)
        y_dir2 = np.repeat(map_augmented_labels[label], len/2)
        y_dir = np.hstack([y_dir1, y_dir2])
    else:
        y_dir = np.repeat(label, len)
    return y_dir


def GetLabelsFullSequence(subject, config, root_folder):
    all_labels = pd.read_csv(f'{Path(root_folder).parent.as_posix()}/TEST_FULL_SEQUENCE/Labels.csv')
    SCLabels = all_labels.query('Subject == @subject').query('Config == @config')

    l_gt = []
    for index, row in SCLabels.iterrows():
        label = row['Label']
        init_frame = row['InitFrame']
        end_frame = row['EndFrame']
        num_repeats = end_frame - init_frame + 1
        l_gt.extend([label] * num_repeats)

    l_gt = np.array(l_gt)
    return l_gt


def ResizeEventFrames(X, factor):
    return X[:, ::factor, ::factor, :, :]


def ResizeInfraredFrames(X, inf_height, inf_width):
    X_resized = np.zeros((X.shape[0], inf_height, inf_width, 1))
    for j in range(X.shape[0]):
        im = X[j, :, :, :]
        res = cv2.resize(im, dsize=(inf_width, inf_height))
        X_resized[j, :, :, :] = res[:, :, np.newaxis]
    return X_resized


def DataAugmentationEvents(X):
    X_flip = X[:, ::-1, :, :, :]
    X = np.vstack([X, X_flip])
    return X


def DataAugmentationInfrared(X):
    X_flip = X[:, ::-1, :, :]
    X = np.vstack([X, X_flip])
    return X


def StackInfraredImages(X, frames_per_image):
    n_images = X.shape[0] - (frames_per_image - 1)
    X_im = np.zeros((n_images, X.shape[1], X.shape[2], frames_per_image))
    for i in range(n_images):
        for j in range(frames_per_image):
            X_im[i, :, :, j] = X[i + j, :, :, 0]
    return X_im
