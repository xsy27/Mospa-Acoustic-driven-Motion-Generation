import os
import argparse
from essentia.standard import *
import librosa
import numpy as np
import scipy
from data.extractor import FeatureExtractor
from smplx import SMPLX
import torch
from tqdm import tqdm
from utils.nn_transforms import aa2mat, aa2repr6d, mat2quat
from data.loader import SAMDataset

parser = argparse.ArgumentParser()

# sam config
parser.add_argument('--dataset_dir', type=str, default='sam')
parser.add_argument('--input_audio_dir', type=str, default='sam/audio')
parser.add_argument('--input_motion_dir', type=str, default='sam')
parser.add_argument('--smpl_dir', type=str, default='smpl_models/smplx')

parser.add_argument('--train_dir', type=str, default='data/sam_train')
parser.add_argument('--val_dir', type=str, default='data/sam_val')
parser.add_argument('--test_dir', type=str, default='data/sam_test')

parser.add_argument('--split_train_file', type=str, default='sam/splits/crossmodal_train.txt')
parser.add_argument('--split_val_file', type=str, default='sam/splits/crossmodal_val.txt')
parser.add_argument('--split_test_file', type=str, default='sam/splits/crossmodal_test.txt')

config = parser.parse_args()

extractor = FeatureExtractor()

if not os.path.exists(config.train_dir):
    os.mkdir(config.train_dir)
if not os.path.exists(config.val_dir):
    os.mkdir(config.val_dir)
if not os.path.exists(config.test_dir):
    os.mkdir(config.test_dir)

split_train_file = config.split_train_file
split_val_file = config.split_val_file
split_test_file = config.split_test_file

njoints = 25

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(config.device)

config.sampling_rate = 30720
config.n_fft = 2048
config.hop_length = 256
config.stft_window = scipy.signal.windows.hann

def make_sam_dataset(sam_dir):
    print('---------- Extract features from raw audio ----------')
    dataset = SAMDataset(sam_dir)

    fnames = []
    num_audios, num_motions = 0, 0
    train, val, test = split_data()

    ii = 0
    all_names = sorted(train + val + test)
    for fname in tqdm(all_names):
        # if ii > 50:
        #     break
        # ii += 1
        audio_file_l = os.path.join(dataset.audio_dir, fname + '_l.wav')
        audio_file_r = os.path.join(dataset.audio_dir, fname + '_r.wav')
        
        if (fname not in train) and (fname not in val) and (fname not in test):
            print(f'Not in set!')
            continue

        if fname in fnames:
            print(f'Already scaned!')
            continue

        sr = config.sampling_rate
        n_fft = config.n_fft
        hop_length = config.hop_length
        window = config.stft_window

        fnames.append(fname)
        
        ### load audio features ###
        y_l, sr_l = librosa.load(path=audio_file_l, sr=sr)
        y_r, sr_r = librosa.load(path=audio_file_r, sr=sr)
        assert (sr_l==sr_r==sr)

        audio_feature_l = extract_acoustic_feature(y_l, sr, n_fft, hop_length, window)
        audio_feature_r = extract_acoustic_feature(y_r, sr, n_fft, hop_length, window)
        audio_feature = np.concatenate([audio_feature_l, audio_feature_r], axis=1)
        num_audios += 1

        ### load pose sequence ###
        smpl_trans, smpl_poses, smpl_betas, sound_sources = dataset.load_motion(fname)
        nframes = smpl_poses.shape[0]
        smpl = None
        smpl = SMPLX(model_path='smpl_models/smplx',
                    use_face_contour=False,
                    gender='NEUTRAL_2020', 
                    num_betas=300,
                    num_expression_coeffs=100,
                    use_pca=False,
                    batch_size=nframes).eval().to(config.device)
        motion_vec, ssl = extract_pose_feature(smpl, smpl_trans, smpl_poses, smpl_betas=smpl_betas, sound_sources=sound_sources)
        num_motions += 1

        # original 120 fps -> 30 fps
        new_audio, new_motion, new_ssl = align_data(audio_feature.tolist(), motion_vec.tolist(), ssl.tolist())
        del_frames = [i for i in range(len(new_motion)) if i%4!=0]
        new_audio = np.delete(new_audio, del_frames, axis=0)
        new_motion = np.delete(new_motion, del_frames, axis=0)
        new_ssl = np.delete(new_ssl, del_frames, axis=0)

        # save features into npz files
        data_class = "train" if fname in train else "val" if fname in val else "test"

        save_data_npz(np.array(new_audio), np.array(new_motion), np.array(new_ssl), fname, data_class, smpl_betas=smpl_betas)
        if ii == 0 or ii == 1:
            print("audio -->", audio_feature.shape)
            print("motion -->", motion_vec.shape)
            print("ssl -->", ssl.shape)
            print("new audio -->", f"({len(new_audio)}, {len(new_audio[0])})")
            print("new motion -->", f"({len(new_motion)}, {len(new_motion[0])})")
            print("new ssl -->", f"({len(new_ssl)}, {len(new_ssl[0])})")
            ii += 1

    assert num_audios == num_motions
    assert len(fnames) == num_motions
    return

def extract_pose_feature(smpl, smpl_trans, smpl_poses, smpl_betas=None, sound_sources=None):
    nframes = smpl_poses.shape[0]
    
    smpl_poses = torch.from_numpy(smpl_poses).float().to(config.device)
    smpl_trans = torch.from_numpy(smpl_trans).float().to(config.device)
    smpl_betas = torch.from_numpy(smpl_betas).reshape(1, -1).float().to(config.device)
    global_orient = smpl_poses[:, :3]
    body_pose = smpl_poses[:, 3:66]
    jaw_pose = smpl_poses[:, 66:69]
    leye_pose = smpl_poses[:, 69:72]
    reye_pose = smpl_poses[:, 72:75]
    left_hand_pose = smpl_poses[:, 75:120]
    right_hand_pose = smpl_poses[:, 120:]
    body_parms = {
        'global_orient': global_orient,'body_pose': body_pose,
        'jaw_pose': jaw_pose,'leye_pose': leye_pose,
        'reye_pose': reye_pose,'left_hand_pose': left_hand_pose,
        'right_hand_pose': right_hand_pose,'transl': smpl_trans,
        'betas': smpl_betas}
    keypoints = smpl.forward(**body_parms).joints.cpu().detach().numpy()[:, 0:njoints, :]
    smpl_poses = smpl_poses.cpu().detach()
    smpl_trans = smpl_trans.cpu().detach()

    global_shift = keypoints[:, 0:1, :]
    keypoints = keypoints - global_shift
    keypoints = keypoints + smpl_trans.numpy().reshape(nframes, 1, 3)

    smpl_poses = smpl_poses[:, :75]

    # calc pose features
    root_rot_mat = np.array([
        aa2mat(smpl_pose[0:3]).numpy() for smpl_pose in smpl_poses
    ]) #(nframes, 3, 3)
    # root_angular_vel_y = np.concatenate([
    #     y_axis_angular_vel_from_marix(torch.from_numpy(root_rot_mat)).detach().cpu().numpy(),
    #     np.zeros(1)
    # ], axis=0)[..., np.newaxis] #(nframes, 1)

    root_pos = keypoints[:, 0, :].reshape(nframes, -1) #(nframes, 3)

    root_rot_6d = np.array([
        aa2repr6d(smpl_pose[:3].reshape(1, 3)).flatten().numpy() for smpl_pose in smpl_poses
    ]) #(nframes, 6)
    root_vel = np.concatenate([root_pos[1:, :] - root_pos[:-1, :], np.zeros((1, 3))], axis=0)
    root_vel_x = np.concatenate([root_pos[1:, 0] - root_pos[:-1, 0], np.zeros(1)], axis=0)[..., np.newaxis] #(nframes, 1)
    root_vel_z = np.concatenate([root_pos[1:, 2] - root_pos[:-1, 2], np.zeros(1)], axis=0)[..., np.newaxis] #(nframes, 1)
    root_pos_y = root_pos[:, 1][..., np.newaxis] #(nframes, 1)

    keypoints_pos = keypoints[:, 1:njoints, :].reshape(nframes, -1) #(nframes, (njoints-1)*3)
    keypoints_vel = np.concatenate([keypoints_pos[1:] - keypoints_pos[:-1], np.zeros((1, (njoints-1)*3))], axis=0) #(nframes, (njoints-1)*3)
    keypoints_rot_6d = np.array([
        aa2repr6d(smpl_pose[3:].reshape(((njoints-1), 3))).flatten().numpy() for smpl_pose in smpl_poses
    ]) #(nframes, (njoints-1)*6)

    foot_keypoints = keypoints[:, [7, 8, 10, 11], :] #(nframes, 4, 3)
    foot_vels = np.concatenate([np.linalg.norm(foot_keypoints[1:] - foot_keypoints[:-1], axis=2), np.zeros((1, 4))], axis=0) #(nframes, 4)
    threshold = 5e-3
    # foot_contacts = np.where(np.abs(foot_vels)<threshold, 1, 0)

    # pelvis -> spine 1 -> spine 2 -> spine 3 -> neck -> head
    # 0 -> 3 -> 6 -> 9 -> 12 -> 15
    joints_rot_mats = np.array(aa2mat(smpl_poses.reshape(-1, njoints, 3)))
    root2head_rot_mats = np.array([joints_rot_mat[3]@joints_rot_mat[6]@joints_rot_mat[9]@joints_rot_mat[12]@joints_rot_mat[15]
                                    for joints_rot_mat in joints_rot_mats])
    global_head_orients = np.array([joints_rot_mat[0] @ root2head_rot_mat
                                    for joints_rot_mat, root2head_rot_mat in zip(joints_rot_mats, root2head_rot_mats)])
    local_head_pos = keypoints[:, 15, :] - keypoints[:, 0, :]

    global_head_orients = mat2quat(torch.from_numpy(global_head_orients)) # w x y z (nframes, 4)
    global_head_orients = global_head_orients.numpy()

    pose_vec = np.concatenate([
        root_pos, # 3
        keypoints_pos, # 3*(njoints-1)
        root_rot_6d, # 6
        keypoints_rot_6d, # 6*(njoints-1)
        root_vel, # 3
        keypoints_vel, # 3*((njoints-1))
    ], axis=1) # 12*njoints

    ssl_features = np.concatenate([
        sound_sources, # 3
    ], axis=1) # 3
    
    return pose_vec, ssl_features

def extract_acoustic_feature(audio, sr, n_fft, hop_length, window):

    # audio = np.mean(np.array([audio_l, audio_r]), axis=0)
    melspe_db = extractor.get_melspectrogram(audio, sr, n_fft, hop_length, window)
    nframes = melspe_db.shape[1]
    
    # features used for motion synthesis
    mfcc = extractor.get_mfcc(melspe_db)
    mfcc_delta = extractor.get_mfcc_delta(mfcc)
    # mfcc_delta2 = extractor.get_mfcc_delta2(mfcc)

    audio_harmonic, audio_percussive = extractor.get_hpss(audio, n_fft, hop_length, window)
    harmonic_melspe_db = extractor.get_harmonic_melspe_db(audio_harmonic, sr, n_fft, hop_length, window)
    percussive_melspe_db = extractor.get_percussive_melspe_db(audio_percussive, sr)
    chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, sr, n_fft, hop_length, octave=7 if sr==15360*2 else 5)
    chroma_cqt = chroma_cqt if chroma_cqt.shape[-1] == mfcc.shape[-1] else chroma_cqt[:, :-1]
    chroma_stft = extractor.get_chroma_stft(audio_harmonic, sr, n_fft, hop_length, window)

    onset_env = extractor.get_onset_strength(audio_percussive, sr, n_fft, hop_length, window)
    tempogram = extractor.get_tempogram(onset_env, sr, hop_length, window)
    beats_one_hot, peaks_one_hot = extractor.get_onset_beat(onset_env, sr, hop_length)
    onset_tempo, onset_beat = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    onset_env = onset_env.reshape(1, -1)

    rms_energy = extractor.get_rms_energy(audio, n_fft, hop_length, window)
    active_frames = rms_energy > 0.01

    audio_feature = np.concatenate([
        # motion synthesis
        mfcc, # 20
        mfcc_delta, # 20
        # harmonic_melspe_db,
        chroma_cqt,
        chroma_stft,
        onset_env, # 1
        tempogram, # 1068
        beats_one_hot, # 1
        rms_energy, # 1
        active_frames, # 1
    ], axis=0) # 

    # return audio_feature.T, binaural_features.T
    return audio_feature.T

def align_data(audio, motion, ssl):
    # print('---------- Align the frames of audio and motion ----------')
    min_seq_len = min(len(audio), len(motion))
    new_audio = [audio[i] for i in range(min_seq_len)]
    new_motion = [motion[i] for i in range(min_seq_len)]
    new_ssl = [ssl[i] for i in range(min_seq_len)]

    return new_audio, new_motion, new_ssl

def save_data_npz(audio, motion, ssl, fname, data_class, smpl_betas=None):
    # print(fname)
    if data_class == "train":
        # print('---------- train data ----------')
        np.savez(os.path.join(config.train_dir, f'{fname}.npz'),
                 id=fname,
                 audio_array=audio,
                 motion_array=motion,
                 ssl_array=ssl,
                 smpl_betas=smpl_betas)
    
    elif data_class == "val":
        # print('---------- val data ----------')
        np.savez(os.path.join(config.val_dir, f'{fname}.npz'),
                 id=fname,
                 audio_array=audio,
                 motion_array=motion,
                 ssl_array=ssl,
                 smpl_betas=smpl_betas)

    else:
        # print('---------- test data ----------')
        np.savez(os.path.join(config.test_dir, f'{fname}.npz'),
                 id=fname,
                 audio_array=audio,
                 motion_array=motion,
                 ssl_array=ssl,
                 smpl_betas=smpl_betas)

def split_data():
    train, val, test = [], [], []

    # print('---------- Split data into train, val and test ----------')
    
    train_file = open(split_train_file, 'r')
    for fname in train_file.readlines():
        train.append(fname.strip())
    train_file.close()

    val_file = open(split_val_file, 'r')
    for fname in val_file.readlines():
        val.append(fname.strip())
    val_file.close()

    test_file = open(split_test_file, 'r')
    for fname in test_file.readlines():
        test.append(fname.strip())
    test_file.close()

    return train, val, test

def main():
    make_sam_dataset(config.dataset_dir)

if __name__ == '__main__':
    main()