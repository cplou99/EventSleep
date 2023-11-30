"""
Created on Mar 2023
@author: 
@project: EventSleep
"""

import torch, torchvision
from torchvision.models import ResNet18_Weights, ResNet34_Weights
from torch.utils.data import TensorDataset, DataLoader
from data_tools import *
from utils import *

from events_to_frames import npyclipsevents_to_npyclipsframes, aedatevents_to_npyframes
from laplace import Laplace

from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sacred import Experiment

import os, glob
import time
import csv

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

ex = Experiment('ClassificationDeepSleep')

@ex.config
def toy_data():
    toy_data = False

@ex.config
def create_folder(toy_data):
    folder_name = 'baseline'

    # configs_train = [1]
    configs_train = [1, 2, 3]

    if configs_train == [1, 2, 3]:
        folder_name_configs = 'TrainAllConfigs'
    else:
        folder_name_configs = f'TrainConfig{configs_train[0]}'

    checkpoint_paths = sorted(glob.glob(f'./Models/ResNet-E/{folder_name_configs}/{folder_name}/checkpoint*.pth'))
    if len(checkpoint_paths) == 0:
        print("The introduced checkpoint path does not exist")

    if toy_data: root_dir = f'{Path(os.getcwd())}/Toy_Data/'
    else: root_dir = f'{Path(os.getcwd())}/DATA/'

    folder_path = Path(checkpoint_paths[0]).parent.as_posix()


@ex.config
def events_to_frames_pre_processing(root_dir):
    k = 1
    chunk_len_ms = 150
    max_time = 512
    ev_height, ev_width = 480, 640
    np_float = 16
    step_pixels = 2

    event_frame_folder = f'{root_dir}/EventFrames/TEST'

@ex.config
def frames_post_processing():
    data_augmentation = True
    crop_bed = True


@ex.config
def labels_dicts():
    labels_dict = LabelsNames()
    labels_id = list(labels_dict.keys())
    labels_names = list(labels_dict.values())


@ex.config
def model_hyperparameters(k, checkpoint_paths):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_pretrained_weights = True
    save_checkpoints = True

    load_trained_weights = False

    in_channels = 2 * k
    out_channels = 10

    ck = torch.load(checkpoint_paths[0])
    n_epochs = ck['epoch']
    lr = ck['lr']
    batch_size = ck['batch_size']
    weight_decay = ck['weight_decay']
    del ck

    weights_labels = GiveWeightsToLabels()
    metric = torch.nn.CrossEntropyLoss(weight=weights_labels.to(device))

@ex.config
def bayesian_config(checkpoint_paths):
    bayesian = True
    if bayesian:
        n_ens = len(checkpoint_paths)
        laplace = True
    else:
        n_ens = 1
        laplace = False

@ex.config
def training_test_strategy(toy_data, configs_train, batch_size):
    configs_test = [1, 2, 3]
    if toy_data:
        subjects_train = [5]
        subjects_test = [9]
        subjects_full_test = [9]
    else:
        subjects_train = [1, 2, 3, 4, 5, 6, 7, 8, 10]
        subjects_test = [9, 12, 13, 14]
        subjects_full_test = [9, 12, 13, 14]


@ex.capture
def extract_events_data(event_frame_folder, subjects, configs, batch_size, crop_bed, data_augmentation,
                        step_pixels, shuffle, device, chunk_len_ms, k, max_time, toy_data, full_sequence):
    X, y = None, None
    if full_sequence:
        config = configs[0]
        subject = subjects[0]
        sc_file = f'{Path(event_frame_folder).parent.as_posix()}/TEST_FULL_SEQUENCE/subject{subject:02}_config{config}.npy'
        if not os.path.exists(sc_file):
            aedatevents_to_npyframes(subject, config, chunk_len_ms, k, max_time, toy_data)
        dirs = [sc_file]
    else:
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

        if full_sequence:
            y_dir = GetLabelsFullSequence(subject, config, event_frame_folder)
        else:
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
def inference_model(model, test_loader):
    y_pred, y_gt, outputs = [], [], []
    t0 = time.time()
    for inputs, targets in test_loader:
        output = model(inputs)
        y_pred.append(output.argmax(-1).detach().cpu().numpy().tolist())
        y_gt.append(targets.detach().cpu().numpy().tolist())
        outputs.append(output.cpu().detach())

    l_gt = [item for sublist in y_gt for item in sublist]
    l_pred = [item for sublist in y_pred for item in sublist]
    outputs = torch.cat(outputs)
    probs = torch.softmax(outputs, dim=1)
    t1 = time.time()
    print("Time spent during inference:", t1 - t0)
    return l_gt, l_pred, probs

@ex.capture
def fit_laplace_classifier(model, train_loader, batch_size, device):
    y_train, feats_train = [], []
    for inputs, targets in train_loader:
        outs = model.resnet(inputs.to('cuda:0'))
        y_train.append(targets.detach().cpu().numpy().tolist())
        feats_train.append(outs.detach().cpu())

    feats_train = torch.cat(feats_train)
    y_train = np.array([item for sublist in y_train for item in sublist])
    X, y = feats_train.to(device), torch.from_numpy(y_train).to(device)
    dataset = TensorDataset(X, y)
    feat_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    t0 = time.time()
    lap_classifier = Laplace(model.fc, 'classification', subset_of_weights='all', hessian_structure='full')
    lap_classifier.fit(feat_loader)
    t1 = time.time()
    print("Time fit Laplace", t1 - t0)
    return lap_classifier


@ex.capture
def inference_laplace_model(model, lap_classifier, test_loader, device):
    y_pred, y_gt, probs = [], [], []
    t0 = time.time()
    for inputs, targets in test_loader:
        feats = model.resnet(inputs)
        output = lap_classifier(feats.detach(), pred_type='glm', link_approx='bridge')
        y_pred.append(output.argmax(-1).detach().cpu().numpy().tolist())
        y_gt.append(targets.detach().cpu().numpy().tolist())
        probs.append(output.cpu().detach().numpy().tolist())

    l_gt = [item for sublist in y_gt for item in sublist]
    l_pred = [item for sublist in y_pred for item in sublist]
    probs = [item for sublist in probs for item in sublist]

    t1 = time.time()
    print("Time spent during inference with Laplace:", t1 - t0)
    return l_gt, l_pred, probs

# @ex.capture
# def accuracy_results(l_gt, l_pred, probs, out_channels, configs_test, folder_path, model_name):
#     matrix = confusion_matrix(l_gt, l_pred, labels=np.arange(0, out_channels))
#     plot_cfm(folder_path=folder_path, matrix=matrix, title=f'{model_name}_FramesPred_Config{configs_test}')
#
#     clips_gt, clips_pred = mode_pred_per_clips(l_gt, l_pred)
#     clip_matrix = confusion_matrix(clips_gt, clips_pred, labels=np.arange(0, out_channels))
#     plot_cfm(folder_path=folder_path, matrix=clip_matrix, title=f'{model_name}_ClipPred_Config{configs_test}')
#
#     clips_gt, bayesian_clips_pred, bayesian_clips_output = prob_pred_per_clips(l_gt, probs)
#     bayesian_clip_matrix = confusion_matrix(clips_gt, bayesian_clips_pred, labels=np.arange(0, out_channels))
#     plot_cfm(folder_path=folder_path, matrix=bayesian_clip_matrix, title=f'{model_name}_ProbClipPred_Config{configs_test}')

@ex.capture
def test_model(folder_path, model, train_loader, checkpoint_paths, subjects_test, configs_test, batch_size, laplace,
               out_channels, n_ens):
    t0 = time.time()
    test_loader = extract_events_data(subjects=subjects_test, configs=configs_test, batch_size=batch_size,
                                      data_augmentation=False, shuffle=False, full_sequence=False)
    t1 = time.time()
    print("Time spent loading test data:", t1 - t0)

    all_ens_probs, all_lapens_probs = [], []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Perform inference
        l_gt, l_pred, probs = inference_model(model, test_loader)
        all_ens_probs.append(probs)

        if laplace:
            lap_classifier = fit_laplace_classifier(model, train_loader)
            lap_gt, lap_pred, lap_probs = inference_laplace_model(model, lap_classifier, test_loader)
            all_lapens_probs.append(lap_probs)

        clips_gt, clips_pred = mode_pred_per_clips(l_gt, l_pred)
        acc = balanced_accuracy_score(clips_gt, clips_pred)
        print("Checkpoint:", checkpoint_path, "accuracy", acc)
        clip_matrix = confusion_matrix(clips_gt, clips_pred, labels=np.arange(0, 10))
        print(clip_matrix)

    all_ens_probs = np.stack(all_ens_probs)
    ens_probs = np.mean(all_ens_probs, axis=0)
    ens_preds = np.argmax(ens_probs, axis=-1)

    if laplace:
        all_lapens_probs = np.stack(all_lapens_probs)
        lapens_probs = np.mean(all_lapens_probs, axis=0)
        lapens_preds = np.argmax(lapens_probs, axis=-1)

    # Accuracy results
    accuracy_results(l_gt, ens_preds, ens_probs, out_channels, configs_test, folder_path, model_name='Determ')
    if laplace:
        accuracy_results(lap_gt, lapens_preds, lapens_probs, model_name='Laplace')

    # Calibration results
    ace, ece, mce = CalibrationMetrics(ens_probs, np.array(l_gt))
    print("[Determ. Classifier] ECE:", ece, "MCE:", mce, "ACE:", ace)
    PlotReliabilityDiagram(ens_probs, np.array(l_gt), f'Determ_Config{configs_test}', root=folder_path)

    if laplace:
        ace, ece, mce = CalibrationMetrics(lapens_probs, np.array(lap_gt))
        print("[Laplace Classifier] ECE:", ece, "MCE:", mce, "ACE:", ace)
        PlotReliabilityDiagram(lapens_probs, np.array(lap_gt), f'Laplace_Config{configs_test}', root=folder_path)

        # Plot histograms to compare
        idxs = np.random.randint(0, len(l_gt), 15)
        plot_histograms(ens_probs, lapens_probs, l_gt, test_loader.dataset.tensors[0], n_ens, idxs, root=folder_path)

    print("Inference finished")
    print("Time spent loading data:", t1 - t0)

    return


@ex.capture
def test_full_sequence_model(model, train_loader, checkpoint_paths, subject_full_test, config, batch_size, laplace,
                             out_channels, folder_path):

    y_pred, y_gt, outputs = [], [], []
    test_loader = extract_events_data(subjects=[subject_full_test], configs=[config], batch_size=batch_size,
                                     data_augmentation=False, shuffle=False, full_sequence=True)

    all_ens_probs, all_lapens_probs = [], []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Perform inference
        l_gt, l_pred, probs = inference_model(model, test_loader)
        all_ens_probs.append(probs)

        if laplace:
            lap_classifier = fit_laplace_classifier(model, train_loader)
            lap_gt, lap_pred, lap_probs = inference_laplace_model(model, lap_classifier, test_loader)
            all_lapens_probs.append(lap_probs)

    all_ens_probs = np.stack(all_ens_probs)
    ens_probs = np.mean(all_ens_probs, axis=0)
    ens_preds = np.argmax(ens_probs, axis=-1)

    if laplace:
        all_lapens_probs = np.stack(all_lapens_probs)
        lapens_probs = np.mean(all_lapens_probs, axis=0)
        lapens_preds = np.argmax(lapens_probs, axis=-1)

    matrix = confusion_matrix(l_gt, ens_preds, labels=np.arange(0, out_channels))
    plot_cfm(folder_path=folder_path, matrix=matrix, title=f'ens_fullsequence_s{subject_full_test}_c{config}')

    matrix = confusion_matrix(lap_gt, lapens_preds, labels=np.arange(0, out_channels))
    plot_cfm(folder_path=folder_path, matrix=matrix, title=f'lapens_fullsequence_s{subject_full_test}_c{config}')

    file_name = f'{folder_path}/results/fullsequences_prediction.csv'
    file_exists = os.path.isfile(file_name)
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["subject", "config", "n_frame", "y_gt", "y_pred"])
        for j in range(len(l_gt)):
            writer.writerow([subject_full_test, config, j, l_gt[j], l_pred[j]])
    return


@ex.automain
def run(folder_path, subjects_test, subjects_full_test, subjects_train, configs_train, root_dir):
    ex.commands["print_config"]()

    results_path = f'{folder_path}/results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # We require the training set to fit Laplace approximation
    train_loader = extract_events_data(event_frame_folder=f'{root_dir}/EventFrames/TRAIN', subjects=subjects_train,
                                       configs=configs_train, shuffle=True,  full_sequence=False)

    model, optimizer = load_model(load_pretrained_weights=False)
    # test_model(model=model, subjects_test=subjects_test, train_loader=train_loader)

    for subject_full_test in subjects_full_test:
        for config in [1, 2, 3]:
            test_full_sequence_model(model=model, train_loader=train_loader, subject_full_test=subject_full_test, config=config)

    ex.commands["save_config"](config_filename=f'{folder_path}/test_details.json')
