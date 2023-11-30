import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from tabulate import tabulate
from scipy import stats
import os
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import pandas as pd

class MyResNet(torch.nn.Module):
    def __init__(self, resnet_model, in_channels, out_channels):
        super(MyResNet, self).__init__()
        resnet_model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.pretrained = resnet_model
        self.fc = torch.nn.Linear(1000, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.xavier_normal(self.fc.weight)

    def resnet(self, x):
        x = self.pretrained(x)
        return x

    def forward(self, x):
        x = self.pretrained(x)
        x = self.fc(x)
        return x

class MyResNetIR(torch.nn.Module):
    def __init__(self, resnet_model, in_channels, out_channels):
        super(MyResNetIR, self).__init__()
        resnet_model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet_model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                             bias=False)
        self.pretrained = resnet_model
        self.extra_layer = torch.nn.Sequential(torch.nn.Linear(1000, out_channels))

    def forward(self, x):
        x = self.pretrained(x)
        x = self.extra_layer(x)
        return x

def accuracy_results(l_gt, l_pred, probs, out_channels, configs_test, folder_path, model_name):
    matrix = confusion_matrix(l_gt, l_pred, labels=np.arange(0, out_channels))
    plot_cfm(folder_path=folder_path, matrix=matrix, title=f'{model_name}_FramesPred_Config{configs_test}')

    clips_gt, clips_pred = mode_pred_per_clips(l_gt, l_pred)
    clip_matrix = confusion_matrix(clips_gt, clips_pred, labels=np.arange(0, out_channels))
    plot_cfm(folder_path=folder_path, matrix=clip_matrix, title=f'{model_name}_ClipPred_Config{configs_test}')

    clips_gt, bayesian_clips_pred, bayesian_clips_output = prob_pred_per_clips(l_gt, probs)
    bayesian_clip_matrix = confusion_matrix(clips_gt, bayesian_clips_pred, labels=np.arange(0, out_channels))
    plot_cfm(folder_path=folder_path, matrix=bayesian_clip_matrix, title=f'{model_name}_ProbClipPred_Config{configs_test}')


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