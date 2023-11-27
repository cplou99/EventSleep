"""
Created on Mar 2023
@author: 
@project: EventSleep
"""

from pathlib import Path

# First import library
# Import Numpy for easy array manipulation
from tqdm import tqdm
from fastai.imports import *
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from data_tools import LabelsNames, CropBed
from events_to_frames import aedatevents_to_npyframes, npyclipsevents_to_npyclipsframes
from data_tools import TrainOrTest

toy_data = True
if toy_data:
    root_dir = './Toy_Data'
else:
    root_dir = f'{Path(os.getcwd()).parent.as_posix()}/DATA'

subject = 9
config = 1

full_sequence = True
clip_num = 0

crop_bed = True

if full_sequence:
    f_name_ir = f'{root_dir}/Infrared/TEST_FULL_SEQUENCE/subject{subject:02}_config{config}.mp4'
    f_name_labels_ir = f'{root_dir}/Infrared/TEST_FULL_SEQUENCE/Labels.csv'
    all_labels_infrared = pd.read_csv(f_name_labels_ir)
    SCLabels_infrared = all_labels_infrared.query('Subject == @subject').query('Config == @config')

    f_name_event_frames = f'{root_dir}/EventFrames/TEST_FULL_SEQUENCE/subject{subject:02}_config{config}.npy'
    if not Path(f_name_event_frames).exists():
        aedatevents_to_npyframes(subject, config, toy_data=toy_data)
    f_name_labels_ev = f'{root_dir}/EventFrames/TEST_FULL_SEQUENCE/Labels.csv'
    all_labels_events = pd.read_csv(f_name_labels_ev)
    SCLabels_events = all_labels_events.query('Subject == @subject').query('Config == @config')

    Labels_dict = LabelsNames()


    event_frames = np.load(f_name_event_frames)
    ir_cap = cv2.VideoCapture(f_name_ir)
    ir_fps = ir_cap.get(cv2.CAP_PROP_FPS)

    ############################################################
    # Video set up
    ############################################################
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs = axs.ravel()
    fig.subplots_adjust(top=1.1, bottom=0.0)

    if crop_bed:
        img_ir = axs[0].imshow(np.random.rand(360, 500, 3))
        img_events = axs[1].imshow(np.random.rand(360, 500), cmap='Reds')

    else:
        img_ir = axs[0].imshow(np.random.rand(400, 700, 3))
        img_events = axs[1].imshow(np.random.rand(480, 640), cmap='Reds')

    axs[0].set_title('Infrared')
    axs[1].set_title('Event Frames')
    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)

    writer = animation.FFMpegWriter(fps=ir_fps)
    video_dir = f'./Renders/Full_Sequences'
    if not os.path.exists(video_dir):  os.makedirs(video_dir)
    video_filename = f'{video_dir}/subject{subject:02}_config{config}.mp4'
    writer.setup(fig, video_filename, dpi=200)


    infrared_n_frames = SCLabels_infrared.iloc[-1]['EndFrame']
    event_n_frames = SCLabels_events.iloc[-1]['EndFrame']
    ratio = infrared_n_frames/event_n_frames


    n_frame_ir = 0

    pbar = tqdm()

    # Streaming loop
    for n_frame_ev in range(event_frames.shape[0]):
        event_frame = event_frames[n_frame_ev, :, :, :, :]

        # Get IR frame
        if n_frame_ir <= int(n_frame_ev * ratio):
            ir_frame_exists, ir_frame = ir_cap.read()
            if ir_frame_exists:
                ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2RGB)
            else:
                break
            n_frame_ir += 1
            if crop_bed: ir_frame = CropBed(ir_frame, 'Infrared', subject)

        # 1 - Plot IR
        img_ir.set_array(ir_frame)

        # 2 - Plot event_frame
        if crop_bed: event_frame = CropBed(event_frame, 'Events', subject)

        # white_image = np.ones((360, 500, 3))
        axs[1].clear()
        axs[1].set_title('Event Frames')
        axs[1].get_xaxis().set_visible(False)
        axs[1].get_yaxis().set_visible(False)

        m_pos = event_frame[:, :, 0, 0]
        m_pos[m_pos == 0] = np.nan
        m_pos = m_pos.astype('float')
        axs[1].imshow(m_pos, alpha=0.5, cmap='Reds')

        m_neg = event_frame[:, :, 0, 1]
        m_neg[m_neg == 0] = np.nan
        m_neg = m_neg.astype('float')
        axs[1].imshow(m_neg, alpha=0.5, cmap='Greens')


        # Plot Labels
        row = SCLabels_events[(SCLabels_events['InitFrame'] <= n_frame_ev).values * (n_frame_ev < SCLabels_events['EndFrame']).values]
        if len(row) != 0:
            l_id = row['Label'].values[0]
            l_id = int(l_id)
            l_name = Labels_dict[l_id]
            fig.suptitle(f'{l_id:02} || {l_name}', fontsize=20)


        writer.grab_frame()
        pbar.update()

    # %%

    writer.finish()
    ir_cap.release()

else:
    train_or_test = TrainOrTest(subject)

    f_name_ir = glob.glob(f'{root_dir}/Infrared/{train_or_test}/subject{subject:02}_config{config}/clip{clip_num:02}*')[0]
    f_name_event_frames = glob.glob(f'{root_dir}/EventFrames/{train_or_test}/subject{subject:02}_config{config}/clip{clip_num:02}*.npy')
    if len(f_name_event_frames) == 0:
        npyclipsevents_to_npyclipsframes(subject, config, toy_data=toy_data)

    f_name_event_frames = glob.glob(f'{root_dir}/EventFrames/{train_or_test}/subject{subject:02}_config{config}/clip{clip_num:02}*.npy')[0]
    l_id = int(f_name_ir[-1])
    filename = f_name_ir.split('/')[-1]

    Labels_dict = LabelsNames()
    event_frames = np.load(f_name_event_frames)

    frames = []  # List to hold all frames
    frames_paths = sorted(glob.glob(f'{f_name_ir}/*'))
    for frame_path in frames_paths:
        frame = cv2.imread(frame_path)
        frames.append(frame)
    infrared_frames = np.stack(frames)

    ############################################################
    # Video set up
    ############################################################
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs = axs.ravel()
    fig.subplots_adjust(top=1.1, bottom=0.0)

    if crop_bed:
        img_ir = axs[0].imshow(np.random.rand(360, 500, 3))
        img_events = axs[1].imshow(np.random.rand(360, 500), cmap='Reds')

    else:
        img_ir = axs[0].imshow(np.random.rand(400, 700, 3))
        img_events = axs[1].imshow(np.random.rand(480, 640), cmap='Reds')

    axs[0].set_title('Infrared')
    axs[1].set_title('Event Frames')
    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)

    writer = animation.FFMpegWriter(fps=6)
    video_dir = f'./Renders/Clips/subject{subject:02}_config{config}'
    if not os.path.exists(video_dir):  os.makedirs(video_dir)

    video_filename = f'{video_dir}/{filename}.mp4'
    writer.setup(fig, video_filename, dpi=200)
    infrared_n_frames = infrared_frames.shape[0]
    event_n_frames = event_frames.shape[0]
    ratio = infrared_n_frames / event_n_frames

    n_frame_ir = 0

    pbar = tqdm()

    # Streaming loop
    for n_frame_ev in range(event_frames.shape[0]):
        event_frame = event_frames[n_frame_ev, :, :, :, :]

        # Get IR frame
        if n_frame_ir <= int(n_frame_ev * ratio):
            ir_frame = infrared_frames[n_frame_ir, :]
            n_frame_ir += 1
            if crop_bed: ir_frame = CropBed(ir_frame, 'Infrared', subject)

        # 1 - Plot IR
        img_ir.set_array(ir_frame)

        # 2 - Plot event_frame
        if crop_bed: event_frame = CropBed(event_frame, 'Events', subject)

        # white_image = np.ones((360, 500, 3))
        axs[1].clear()
        axs[1].set_title('Event Frames')
        axs[1].get_xaxis().set_visible(False)
        axs[1].get_yaxis().set_visible(False)

        m_pos = event_frame[:, :, 0, 0]
        m_pos[m_pos == 0] = np.nan
        m_pos = m_pos.astype('float')
        axs[1].imshow(m_pos, alpha=0.5, cmap='Reds')

        m_neg = event_frame[:, :, 0, 1]
        m_neg[m_neg == 0] = np.nan
        m_neg = m_neg.astype('float')
        axs[1].imshow(m_neg, alpha=0.5, cmap='Greens')

        # Plot Labels
        l_id = int(l_id)
        l_name = Labels_dict[l_id]
        fig.suptitle(f'{l_id:02} || {l_name}', fontsize=20)

        writer.grab_frame()
        pbar.update()

    # %%

    writer.finish()