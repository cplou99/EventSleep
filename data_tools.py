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



def mode_pred_per_clips(labels_gt, labels_pred):
    sublists_gt, sublists_pred = [], []  # List to store sublists
    current_sequence_gt, current_sequence_pred = [], []  # List to store the current sequence
    # Iterate through the input list and split it into sublists
    for i in range(len(labels_gt)):
        if i == 0 or labels_gt[i] == labels_gt[i - 1]:
            current_sequence_gt.append(labels_gt[i])
            current_sequence_pred.append(labels_pred[i])
        else:
            sublists_gt.append(current_sequence_gt)
            sublists_pred.append(current_sequence_pred)
            current_sequence_gt, current_sequence_pred = [labels_gt[i]], [labels_pred[i]]

    sublists_gt.append(current_sequence_gt)
    sublists_pred.append(current_sequence_pred)
    n_clips = len(sublists_gt)
    # print(f'Sequence of {n_clips} clips')
    clips_gt, clips_pred = [], []
    for j in range(n_clips):
        clip_preds = sublists_pred[j]
        clip_pred = stats.mode(clip_preds)[0][0]
        clips_gt.append(sublists_gt[j][0])
        clips_pred.append(clip_pred)

    return clips_gt, clips_pred


def prob_pred_per_clips(labels_gt, labels_pred, num_frames=False):
    import scipy
    sublists_gt, sublists_pred, clips_len = [], [], [] # List to store sublists
    current_sequence_gt, current_sequence_pred = [], np.zeros(10) # List to store the current sequence
    # Iterate through the input list and split it into sublists
    for i in range(len(labels_gt)):
        if i == 0 or labels_gt[i] == labels_gt[i - 1]:
            current_sequence_gt.append(labels_gt[i])
            current_sequence_pred += np.array(labels_pred[i])
        else:
            sublists_gt.append(current_sequence_gt)
            sublists_pred.append(current_sequence_pred)
            current_sequence_gt, current_sequence_pred = [labels_gt[i]], np.array(labels_pred[i])

    sublists_gt.append(current_sequence_gt)
    sublists_pred.append(current_sequence_pred)
    n_clips = len(sublists_gt)
    # print(f'Sequence of {n_clips} clips')
    clips_gt, clips_pred, clips_outputs = [], [], []
    for j in range(n_clips):
        clip_preds = sublists_pred[j]
        clip_pred = np.array(clip_preds).argmax()
        clips_len.append(len(sublists_gt[j]))
        clips_gt.append(sublists_gt[j][0])
        clips_pred.append(clip_pred)
        clips_outputs.append(scipy.special.softmax(np.array(clip_preds)))
    clips_outputs = np.array(clips_outputs)
    if num_frames:
        return clips_gt, clips_pred, clips_outputs, clips_len
    else:
        return clips_gt, clips_pred, clips_outputs


def plot_cfm(folder_path, matrix, title):
    labels_names = ['Head', 'Hands', 'RollL', 'RollR',  'LegsS', 'ArmsS', 'LieL', 'LieR', 'LieU', 'LieD']
    sn.set(font_scale=1.75)
    size_title = 22
    x_sub, y_sub = 5, 11.3

    rows_with_headers = [[label] + row.tolist() for label, row in zip(labels_names, matrix)]
    print(tabulate(rows_with_headers, headers=['True \ Predicted'] + labels_names, tablefmt='fancy_grid',
                   numalign='center'))

    acc_weighted = matrix.diagonal().sum() / matrix.sum()
    acc_l = matrix.diagonal() / matrix.sum(axis=1)
    acc = np.nanmean(acc_l)
    is_nan = any(np.isnan(acc_l))
    print(f'Acc_weighted: {acc_weighted:0.2f}, Acc:{acc}')

    acc_l = acc_l.tolist()
    acc_l_s = f'[{acc_l[0]:0.2f}, {acc_l[1]:0.2f}, {acc_l[2]:0.2f}, {acc_l[3]:0.2f}, {acc_l[4]:0.2f}, {acc_l[5]:0.2f}, {acc_l[6]:0.2f}, {acc_l[7]:0.2f}, {acc_l[8]:0.2f}, {acc_l[9]:0.2f}]'
    print(f'Accuracy {acc:0.2f}: {acc_l_s}')

    cfm_title = f'Accuracy {acc:0.2f}: {acc_l_s}'
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 11))
    matrix = matrix / matrix.sum(axis=1)[:, np.newaxis]
    df_cfm = pd.DataFrame(matrix, index=labels_names, columns=labels_names)
    sn.heatmap(df_cfm, vmin=0, vmax=1, annot=True, fmt='.2f', ax=ax)
    ax.set_title(cfm_title, size=size_title)
    ax.set_ylabel('Ground truth')
    ax.set_xlabel('Predicted')
    if is_nan:
        ax.annotate("* Nan means label not present in this experiment", xy=(2.5, 2.5), xytext=(x_sub, y_sub), fontsize=18,
                    ha='center', va='bottom', color='black')
    plt.savefig(f'{folder_path}/results/cfm_{title}.png')



def ReliabilityBins(probs, targets):
    p_ = probs.max(-1)
    N = probs.shape[0]
    bins = list(np.linspace(0, 1, 11))
    confidences, accuracies, count = [], [], []
    for i in range(len(bins) - 1):
        mask = (p_ >= bins[i]) & (p_ < bins[i + 1])
        p = p_[mask]
        tick = (probs[mask, :].argmax(-1) == targets[mask])
        dummy = np.array(list(map(float, tick)))
        count.append(0.0 if not p.any() else mask.sum() / N)
        confidences.append(0.0 if not p.any() else p.mean())
        accuracies.append(0.0 if not dummy.any() else dummy.mean())

    return confidences, accuracies, count

def CalibrationMetrics(probs, targets):
    confidences, accuracies, count = ReliabilityBins(probs, targets)
    dif = np.abs(np.asarray(confidences) - np.asarray(accuracies))
    ece, mce = np.mean(dif * np.asarray(count)), np.max(dif)
    bins_ace = np.count_nonzero(count)
    ace = np.sum(dif)/bins_ace

    return ace, ece, mce


def PlotReliabilityDiagram(probs, targets, title, root=None):
    p = probs.max(-1)
    bins = list(np.linspace(0, 1, 11))
    confidences, accuracies, _ = ReliabilityBins(probs, targets)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), facecolor='white')

    ax.grid(True)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.hist(p, bins=bins, color='lightgray', weights=np.ones_like(p) / len(p), label='Frequency', zorder=2)
    ax.set_xticks(ticks=[round(j, 2) for j in bins], labels=[str(round(j, 2)) for j in bins])
    ax.set_yticks(ticks=[round(j, 2) for j in bins], labels=[str(round(j, 2)) for j in bins])
    ax.plot(confidences, accuracies, color='blue', marker='o', label="Accuracy")
    for i in range(len(bins)-1):
        if i == 0:
            ax.plot([confidences[i], confidences[i]], [accuracies[i], confidences[i]], color='red', label="Calibration Error")
        else:
            ax.plot([confidences[i], confidences[i]], [accuracies[i], confidences[i]], color='red')
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    ax.legend()
    plt.savefig(f'{root}/results/reliability_{title}.png')



def plot_histograms(outputs, laplace_outputs, labels, test_x, n_ens, idxs, root):
    len = test_x.shape[0]
    f_outputs = outputs[-len:]
    f_laplace_outputs = laplace_outputs[-len:]
    f_labels = labels[-len:]
    for k in idxs:
        out = f_outputs[k, :].tolist()
        lap_out = f_laplace_outputs[k, :].tolist()

        event_frame_pos = test_x[k, 0, :, :]
        event_frame_neg = test_x[k, 1, :, :]

        # Example histogram data (two lists of 10 values each, representing probability predictions)
        hist_data1 = np.array(out)
        hist_data2 = np.array(lap_out)

        # Labels for the categories
        labels = ['Head', 'Hands', 'RollL', 'RollR',  'Legs', 'Arms', 'LieL', 'LieR', 'LieU', 'LieD']
        # Adjusting the subplot sizes to make the histograms smaller than the main image
        fig, ax = plt.subplots(3, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        # Plot the image
        m1 = event_frame_pos.detach().cpu().numpy()
        m1[m1 == 0] = np.nan
        m1 = m1.astype('float')
        ax[0].imshow(m1, alpha=0.5, cmap='Greens')

        m2 = event_frame_neg.detach().cpu().numpy()
        m2[m2 == 0] = np.nan
        m2 = m2.astype('float')
        ax[0].imshow(m2, alpha=0.5, cmap='Reds')
        ax[0].set_facecolor('white')
        ax[0].grid(False)
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # ax[0].imshow(image_data, cmap='viridis')
        ax[0].set_title("Event Frame")
        # Plot the first histogram
        ax[1].bar(labels, hist_data1, color='blue')
        if n_ens == 1:
            ax[1].set_title("Determ. Classifier")
        else:
            ax[1].set_title("Ensembles")
        ax[1].set_ylabel("Probability")
        ax[1].set_ylim(0, 1)  # Assuming probabilities are between 0 and 1
        ax[1].set_facecolor('white')


        # Plot the second histogram
        ax[2].bar(labels, hist_data2, color='green')
        if n_ens == 1:
            ax[2].set_title("Laplace classifier")
        else:
            ax[2].set_title("Laplace Ensembles")
        ax[2].set_ylabel("Probability")
        ax[2].set_ylim(0, 1)  # Assuming probabilities are between 0 and 1
        ax[2].set_facecolor('white')

        ax[0].set_axis_on()
        plt.tight_layout()

        histograms_path = f'{root}/results/histograms'
        if not os.path.exists(histograms_path):
            os.makedirs(histograms_path)
        plt.savefig(f'{root}/results/histograms/frame{k}_ens{n_ens}.jpg')

    return