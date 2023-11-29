"""
Created on Mar 2023
@author: 
@project: EventSleep
"""

import torch, torchvision
from torchvision.models import ResNet18_Weights, ResNet34_Weights
from torch.utils.data import TensorDataset, DataLoader
from data_tools import *
import imageio.v3 as iio

from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sacred import Experiment

import glob
import os
import time
import csv

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class MyResNet(torch.nn.Module):
    def __init__(self, resnet_model, in_channels, out_channels):
        super(MyResNet, self).__init__()
        resnet_model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.pretrained = resnet_model
        self.extra_layer = torch.nn.Sequential(torch.nn.Linear(1000, out_channels))

    def forward(self, x):
        x = self.pretrained(x)
        x = self.extra_layer(x)
        return x


ex = Experiment('ClassificationDeepSleep')

@ex.config
def toy_data():
    toy_data = True

@ex.config
def create_folder(toy_data):
    folder_name = 'baseline'

    # train_configs = [1]
    train_configs = [1, 2, 3]

    if train_configs == [1, 2, 3]:
        folder_name_configs = 'TrainAllConfigs'
    else:
        folder_name_configs = f'TrainConfig{train_configs[0]}'

    checkpoint_path = glob.glob(f'./Models/ResNet-IR/{folder_name_configs}/{folder_name}/checkpoint*.pth')[0]
    if not os.path.exists(checkpoint_path):
        print("The introduced checkpoint path does not exist")

    if toy_data: root_dir = f'{Path(os.getcwd())}/Toy_Data/'
    else: root_dir = f'{Path(os.getcwd())}/DATA/'

    folder_path = Path(checkpoint_path).parent.as_posix()


@ex.config
def frames_post_processing(root_dir):
    inf_width = 250
    inf_height = 180
    frames_per_image = 3
    infrared_frame_folder = f'{root_dir}/Infrared/TEST'
    data_augmentation = True
    crop_bed = True


@ex.config
def labels_dicts():
    labels_dict = LabelsNames()
    labels_id = list(labels_dict.keys())
    labels_names = list(labels_dict.values())


@ex.config
def model_hyperparameters(frames_per_image):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_pretrained_weights = True

    in_channels = frames_per_image
    out_channels = 10

    n_epochs = 2
    lr = 1e-3
    batch_size = 32
    weight_decay = 0

    weights_labels = GiveWeightsToLabels()
    metric = torch.nn.CrossEntropyLoss(weight=weights_labels.to(device))


@ex.config
def training_test_strategy(toy_data):
    if toy_data:
        subjects_test = [9]
        subjects_full_test = [9]
    else:
        subjects_test = [9, 12, 13, 14]
        subjects_full_test = [9, 13, 15]
    configs_test = [1, 2, 3]

@ex.capture
def extract_infrared_data(infrared_frame_folder, inf_width, inf_height, frames_per_image, subjects, configs,
                          batch_size, crop_bed, data_augmentation,  shuffle, device, full_sequence):

    X, y = None, None

    if full_sequence:
        subject = subjects[0]
        config = configs[0]
        sc_file = f'{Path(infrared_frame_folder).parent.as_posix()}/TEST_FULL_SEQUENCE/subject{subject:02}_config{config}.mp4'
        dirs = [sc_file]
    else:
        dirs = []
        for subject in subjects:
            for c in configs:
                dirs.append(sorted(glob.glob(f'{infrared_frame_folder}/subject{subject:02}_config{c}/clip*_label*')))
        dirs = [item for sublist in dirs for item in sublist]

    for dir in dirs:
        X_dir = None

        if full_sequence:
            vidcap = cv2.VideoCapture(dir)
            success, im = vidcap.read()
            while success:  # save frame as JPEG file
                im = im[:, :, 0]
                im = im[np.newaxis, :, :, np.newaxis]
                if X_dir is None: X_dir = im
                else: X_dir = np.vstack([X_dir, im])
                success, im = vidcap.read()

        else:
            for im_path in sorted(glob.glob(f'{dir}/*.png')):
                im = iio.imread(im_path)
                im = im[:, :, 0]
                im = im[np.newaxis, :, :, np.newaxis]
                if X_dir is None: X_dir = im
                else: X_dir = np.vstack([X_dir, im])

        X_dir = X_dir.astype('float32')

        if crop_bed:
            X_dir = CropBed(X_dir, 'Infrared', subject)

        if inf_width != X_dir.shape[2] or inf_height != X_dir.shape[1]:
            X_dir = ResizeInfraredFrames(X_dir, inf_height, inf_width)

        if X_dir.shape[0] - (frames_per_image - 1) < 0:
            continue
        X_dir = StackInfraredImages(X_dir, frames_per_image)

        if data_augmentation:
            X_dir = DataAugmentationInfrared(X_dir)

        if full_sequence:
            y_dir = GetLabelsFullSequence(subject, config, infrared_frame_folder)
            y_dir = y_dir[frames_per_image - 1:]
        else:
            y_dir = GetLabelFromDirName(dir, X_dir.shape[0], 'Infrared', data_augmentation)

        if X is None:
            X, y = X_dir, y_dir
        else:
            X, y = np.vstack([X, X_dir]), np.hstack([y, y_dir])

    X = X.astype('float32')
    X = X.transpose(0, 3, 1, 2)
    X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


@ex.capture
def load_model(in_channels, out_channels, lr, device, weight_decay, load_pretrained_weights):

    if load_pretrained_weights:
        resnet_model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        resnet_model = torchvision.models.resnet18()

    model = MyResNet(resnet_model, in_channels, out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model.to(device), optimizer


@ex.capture
def test_model(folder_path, model, subjects_test, configs_test, batch_size, out_channels):
    cfms, clip_cfms = [], []
    y_pred, y_gt, outputs = [], [], []

    t0 = time.time()
    test_loader = extract_infrared_data(subjects=subjects_test, configs=configs_test, batch_size=batch_size,
                                        data_augmentation=False, shuffle=False, full_sequence=False)
    t1 = time.time()
    for inputs, targets in test_loader:
        output = model(inputs)
        y_pred.append(output.argmax(-1).detach().cpu().numpy().tolist())
        y_gt.append(targets.detach().cpu().numpy().tolist())
        outputs.append(output.cpu().detach().numpy().tolist())

    l_gt = [item for sublist in y_gt for item in sublist]
    l_pred = [item for sublist in y_pred for item in sublist]
    outputs = torch.cat(outputs)
    probs = torch.softmax(outputs, dim=1)
    t2 = time.time()

    matrix = confusion_matrix(l_gt, l_pred, labels=np.arange(0, out_channels))
    plot_cfm(folder_path=folder_path, matrix=matrix, title=f'FramesPred_Config{configs_test}')
    cfms.append(matrix)

    clips_gt, clips_pred = mode_pred_per_clips(l_gt, l_pred)
    clip_matrix = confusion_matrix(clips_gt, clips_pred, labels=np.arange(0, out_channels))
    plot_cfm(folder_path=folder_path, matrix=clip_matrix, title=f'ClipPred_Config{configs_test}')
    clip_cfms.append(clip_matrix)

    clips_gt, bayesian_clips_pred, bayesian_clips_output = prob_pred_per_clips(l_gt, probs)
    bayesian_clip_matrix = confusion_matrix(clips_gt, bayesian_clips_pred, labels=np.arange(0, out_channels))
    plot_cfm(folder_path=folder_path, matrix=bayesian_clip_matrix, title=f'ProbClipPred_Config{configs_test}')

    print("Inference finished")
    print("Time spent loading data:", t1 - t0)
    print("Time spent during inference:", t2 - t1)
    return



@ex.capture
def online_test_full_sequence(model, subject_full_test, config, batch_size, out_channels, folder_path):

    y_pred, y_gt, outputs = [], [], []
    test_loader = extract_infrared_data(subjects=[subject_full_test], configs=[config], batch_size=batch_size,
                                     data_augmentation=False, shuffle=False, full_sequence=True)
    for inputs, targets in test_loader:
        output = model(inputs)
        y_pred.append(output.argmax(-1).detach().cpu().numpy().tolist())
        y_gt.append(targets.detach().cpu().numpy().tolist())
        outputs.append(torch.softmax(output, dim=1).cpu().detach().numpy().tolist())

    l_gt = [item for sublist in y_gt for item in sublist]
    l_pred = [item for sublist in y_pred for item in sublist]


    if len(l_gt) != len(l_pred):
        print("ERROR")

    file_name = f'{folder_path}/fullsequences_prediction.csv'
    file_exists = os.path.isfile(file_name)
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["subject", "config", "n_frame", "y_gt", "y_pred"])
        for j in range(len(l_gt)):
            writer.writerow([subject_full_test, config, j, l_gt[j], l_pred[j]])

    matrix = confusion_matrix(l_gt, l_pred, labels=np.arange(0, out_channels))
    plot_cfm(folder_path=folder_path, matrix=matrix, title=f'online_fullsequence_s{subject_full_test}_c{config}')
    return


@ex.automain
def run(folder_path, checkpoint_path, subjects_test, subjects_full_test):
    ex.commands["print_config"]()

    checkpoint = torch.load(checkpoint_path)
    model, optimizer = load_model(lr=checkpoint['lr'], weight_decay=checkpoint['weight_decay'], load_pretrained_weights=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    test_model(model=model, subjects_test=subjects_test)
    for subject_full_test in subjects_full_test:
        online_test_full_sequence(model=model, subject_full_test=subject_full_test, config=1)
    ex.commands["save_config"](config_filename=f'{folder_path}/test_details.json')
