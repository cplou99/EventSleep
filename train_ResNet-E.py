"""
Created on Mar 2023
@author: 
@project: EventSleep
"""

import torch, torchvision
from torchvision.models import ResNet18_Weights, ResNet34_Weights
from torch.utils.data import TensorDataset, DataLoader
from data_tools import *
from events_to_frames import npyclipsevents_to_npyclipsframes


from pathlib import Path
import numpy as np
import time
from sklearn.metrics import balanced_accuracy_score
from sacred import Experiment

import glob
import os
import datetime


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


class MyResNet(torch.nn.Module):
    def __init__(self, resnet_model, in_channels, out_channels):
        super(MyResNet, self).__init__()
        resnet_model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                             bias=False)
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


ex = Experiment('ClassificationDeepSleep')

@ex.config
def toy_data():
    toy_data = True

@ex.config
def create_folder(toy_data):
    configs_train = [1, 2, 3]

    if toy_data: root_dir = f'{Path(os.getcwd())}/Toy_Data'
    else: root_dir = f'{Path(os.getcwd()).parent.as_posix()}/DATA'

    if configs_train == [1, 2, 3]:
        folder_name_configs = 'TrainAllConfigs'
    else:
        folder_name_configs = f'TrainConfig{configs_train[0]}'

    file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    folder_path = os.path.join(f'{Path(os.getcwd())}/Models/ResNet-E/{folder_name_configs}/{file_name}')
    if not os.path.exists(folder_path):  os.makedirs(folder_path)



@ex.config
def events_to_frames_pre_processing(root_dir):
    k = 1
    chunk_len_ms = 150
    max_time = 512
    ev_height, ev_width = 480, 640
    np_float = 16
    step_pixels = 2

    event_frame_folder = f'{root_dir}/EventFrames/TRAIN'
    print()

@ex.config
def frames_post_processing():
    data_augmentation = True
    crop_bed = True


@ex.config
def labels_dicts():
    labels_dict = LabelsNames()
    n_labels = 10
    labels_id = list(labels_dict.keys())
    labels_names = list(labels_dict.values())


@ex.config
def model_hyperparameters(k):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_pretrained_weights = True
    save_checkpoints = True

    load_trained_weights = False

    in_channels = 2 * k
    out_channels = 10

    # Hyperparameters to check
    batch_size = 32
    n_epochs = 10
    lr = 1e-3
    weight_decay = 0.01

    weights_labels = GiveWeightsToLabels()
    metric = torch.nn.CrossEntropyLoss(weight=weights_labels.to(device))


@ex.config
def training_test_strategy(toy_data):
    if toy_data:
        subjects_train = [1]
        subjects_val = [11]
    else:
        subjects_train = [1, 2, 3, 4, 5, 6, 7, 8, 10]
        subjects_val = [11]


@ex.capture
def extract_events_data(event_frame_folder, subjects, configs, batch_size, crop_bed, data_augmentation,
                        step_pixels, shuffle, device, chunk_len_ms, k, max_time, toy_data, dirs=None):

    X, y = None, None
    if dirs is None:
        dirs = []
        for subject in subjects:
            for c in configs:
                sc_folder = f'{event_frame_folder}/subject{subject:02}_config{c}'
                if not os.path.exists(sc_folder):
                    npyclipsevents_to_npyclipsframes(subject, c, chunk_len_ms, k, max_time, toy_data)
                dirs.append(glob.glob(f'{event_frame_folder}/subject{subject:02}_config{c}/clip*_label*.npy'))
        dirs = [item for sublist in dirs for item in sublist]

    for dir in dirs:
        X_dir = np.load(dir)

        if crop_bed:
            X_dir = CropBed(X_dir, 'Events', subject)

        if step_pixels is not None:
            X_dir = ResizeEventFrames(X_dir, step_pixels)

        if data_augmentation:
            X_dir = DataAugmentationEvents(X_dir)

        X_dir = X_dir.astype('float32')
        X_dir = X_dir.reshape(X_dir.shape[0], X_dir.shape[1], X_dir.shape[2], X_dir.shape[-1] * X_dir.shape[-2])

        y_dir = GetLabelFromDirName(dir, X_dir.shape[0], 'Events', data_augmentation)

        if X is None:
            X, y = X_dir, y_dir
        else:
            X, y = np.vstack([X, X_dir]), np.hstack([y, y_dir])

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
def train_model(model, optimizer, metric, subjects_train, subjects_val, n_epochs, save_checkpoints, batch_size, lr,
                weight_decay, configs_train, folder_path):

    t0 = time.time()
    val_loader = extract_events_data(subjects=subjects_val, configs=configs_train, shuffle=True)
    training_loader = extract_events_data(subjects=subjects_train, configs=configs_train, batch_size=batch_size, shuffle=True)
    acc_val_max = 0

    t1 = time.time()
    # Train
    for n_ep in range(n_epochs):
        n_batch = 0
        for inputs, targets in training_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = metric(outputs, targets)
            loss.backward()
            optimizer.step()
            n_batch += 1

        acc_train = balanced_accuracy_score(targets.cpu().detach().numpy(), outputs.argmax(-1).cpu().detach().numpy())
        outputs_val, targets_val = [], []
        for in_val, t_val in val_loader:
            out_val = model(in_val)
            outputs_val.append(out_val.cpu().detach())
            targets_val.append(t_val.cpu().detach())
        outputs_val, targets_val = torch.cat(outputs_val), torch.cat(targets_val)
        acc_val = balanced_accuracy_score(targets_val.numpy(), outputs_val.argmax(-1).numpy())
        print('[Epoch %d] Loss_train: %.3f, Acc_train: %.3f ||  Acc_val: %.3f' % (n_ep + 1, loss.item(), acc_train, acc_val))


        if save_checkpoints and acc_val > acc_val_max:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr': lr,
                'batch_size': batch_size,
                'weight_decay': weight_decay,
                'epoch': n_ep + 1,
                'loss': loss
            }
            acc_val_max = acc_val
            checkpoint_file = os.path.join(folder_path, f'checkpoint_accval{round(acc_val, 2)}.pth')

    torch.save(checkpoint, checkpoint_file)
    t2 = time.time()
    t_data = t1 - t0
    t_model = t2 - t1
    print("Model Training Finished")
    print("Time spent loading data:", t_data)
    print("Time spent training model:", t_model)
    return


@ex.automain
def run(folder_path):
    ex.commands["print_config"]()

    model, optimizer = load_model()
    train_model(model=model, optimizer=optimizer)
    ex.commands["save_config"](config_filename=f'{folder_path}/train_details.json')


