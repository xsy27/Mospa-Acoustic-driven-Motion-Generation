import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import copy
import json
from types import SimpleNamespace as Namespace
import blobfile as bf
from tqdm import tqdm
import warnings
import shutil

from utils import common
from network.models import *
from utils.nn_transforms import aa2repr6d, repr6d2aa
from utils.smplx import get_smplx_joints, smplx_FK

warnings.filterwarnings("ignore")

njoints = 25
skeletons, parents = get_smplx_joints(model_path="smpl_models/smplx", gender='NEUTRAL_2020', njoints=njoints)

def extract_audio_features(config, audio_dim, latent_dim=1024, num_layers=4):
    nframes = 240

    audio_extractor = AudioExtractor(audio_dim, latent_dim, num_layers).to(config.device)
    if bf.exists(config.extractor):
        best_model = torch.load(config.extractor)
        audio_extractor.load_state_dict(best_model['audio_extractor_state_dict'])
    else:
        raise ValueError("loadpoint does not exist!!!")
    audio_extractor.eval()

    save_path = os.path.join(config.feat_save, 'audio')
    if os.path.exists(save_path): shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    shape = None

    for fname in tqdm(os.listdir(config.test_data)):
        id = fname.split('.')[0]
        # print(id)
        file = os.path.join(config.test_data, f'{id}.npz')
        with open(file, 'rb') as f:
            data = np.load(f)
            np_audio = data['audio_array']
            np_ssl = data['ssl_array']
        np_audio = np_audio[:nframes]
        np_ssl = np_ssl[:nframes]
        np_ssl = np_ssl / np.linalg.norm(np_ssl, axis=-1, keepdims=True)
        np_genre = np.full((nframes, 1), int(id[-1]))
        # print(np_audio.shape, np_ssl.shape, np_genre.shape)

        audio = torch.from_numpy(np.concatenate([np_audio, np_ssl, np_genre], axis=-1)).to(config.device).float().unsqueeze(0)
        if shape is None: shape = audio.shape
        audio_latent = audio_extractor(audio)[1].detach().cpu().numpy().squeeze()
        np.save(os.path.join(save_path, f'{id}.npy'), audio_latent)
    print("audio: ", shape)

def extract_gt_motion_features(config, motion_dim, latent_dim=1024, num_layers=4):
    nframes = 240

    motion_extractor = MotionExtractor(motion_dim, latent_dim, num_layers).to(config.device)
    if bf.exists(config.extractor):
        best_model = torch.load(config.extractor)
        motion_extractor.load_state_dict(best_model['motion_extractor_state_dict'])
    else:
        raise ValueError("loadpoint does not exist!!!")
    motion_extractor.eval()

    save_path = os.path.join(config.feat_save, 'motion')
    if os.path.exists(save_path): shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    shape = None

    for fname in tqdm(os.listdir(config.test_data)):
        id = fname.split('.')[0]
        # print(id)
        file = os.path.join(config.test_data, f'{id}.npz')
        with open(file, 'rb') as f:
            data = np.load(f)
            np_motion = data['motion_array']
        np_motion = np_motion[:nframes]
        # print(np_motion.shape)

        motion = torch.from_numpy(np_motion).to(config.device).float().unsqueeze(0)
        if shape is None: shape = motion.shape
        motion_latent = motion_extractor(motion)[1].detach().cpu().numpy().squeeze()
        np.save(os.path.join(save_path, f'{id}.npy'), motion_latent)
    print("gt motion: ", shape)

def extract_motion_features(config, model_type, motion_dim, latent_dim=1024, num_layers=4):
    nframes = 240

    motion_extractor = MotionExtractor(motion_dim, latent_dim, num_layers).to(config.device)
    if bf.exists(config.extractor):
        best_model = torch.load(config.extractor)
        motion_extractor.load_state_dict(best_model['motion_extractor_state_dict'])
    else:
        raise ValueError("loadpoint does not exist!!!")
    motion_extractor.eval()

    save_path = os.path.join(config.feat_save, model_type)
    if os.path.exists(save_path): shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    shape = None
    shapes = None

    for fname in tqdm(os.listdir(config.src_dir), desc=model_type):
        id = fname.split('.')[0]
        bs = 1
        # print(id)
        if model_type.startswith('mospa'):
            file = os.path.join(config.src_dir, f'{id}.npy')
            data = np.load(file, allow_pickle=True).item()
            np_position = data['position']
            np_rotation = data['rotation']
            np_position = np_position.reshape(nframes, -1)
            np_rotation = aa2repr6d(torch.from_numpy(np_rotation)).numpy().reshape(nframes, -1)
            np_velocity = np.concatenate([np_position[1:] - np_position[:-1], np.zeros((1, 3*njoints))], axis=0)

        elif model_type.startswith('edge'):
            file = os.path.join(config.src_dir, f'{id}.npz')
            with open(file, 'rb') as f:
                data = np.load(f)
                pred = data['motions']
            pred = torch.from_numpy(pred)
            assert pred.shape == (bs, nframes, 157)
            pred_rotaa = repr6d2aa(pred[..., 7:].reshape(bs, nframes, njoints, 6)).float()
            pred_transl = pred[..., 4:7]
            pred_pos = smplx_FK(skeletons, parents, bs, pred_rotaa, pred_transl).reshape(nframes, 3*njoints)

            np_position = pred_pos
            np_rotation = pred[..., 7:].reshape(nframes, -1)
            np_velocity = np.concatenate([np_position[1:] - np_position[:-1], np.zeros((1, 3*njoints))], axis=0)
        
        elif model_type.startswith('popdg'):
            file = os.path.join(config.src_dir, f'{id}.npz')
            with open(file, 'rb') as f:
                data = np.load(f)
                pred = data['motions']
            pred = torch.from_numpy(pred)
            assert pred.shape == (bs, nframes, 160)
            pred_rotaa = repr6d2aa(pred[..., 3:-7].reshape(bs, nframes, njoints, 6)).float()
            pred_transl = pred[..., :3]
            pred_pos = smplx_FK(skeletons, parents, bs, pred_rotaa, pred_transl).reshape(nframes, 3*njoints)

            np_position = pred_pos
            np_rotation = pred[..., 3:-7].reshape(nframes, -1)
            np_velocity = np.concatenate([np_position[1:] - np_position[:-1], np.zeros((1, 3*njoints))], axis=0)
        
        elif model_type.startswith('lodge'):
            file = os.path.join(config.src_dir, f'{id}.npz')
            with open(file, 'rb') as f:
                data = np.load(f)
                pred = data['pred']
            pred = torch.from_numpy(pred)
            assert pred.shape == (bs, nframes, 157)
            pred_rotaa = repr6d2aa(pred[..., 7:].reshape(bs, nframes, njoints, 6)).float()
            pred_transl = pred[..., 4:7]
            pred_pos = smplx_FK(skeletons, parents, bs, pred_rotaa, pred_transl).reshape(nframes, 3*njoints)

            np_position = pred_pos
            np_rotation = pred[..., 7:].reshape(nframes, -1)
            np_velocity = np.concatenate([np_position[1:] - np_position[:-1], np.zeros((1, 3*njoints))], axis=0)
        
        elif model_type.startswith('bailando'):
            file = os.path.join(config.src_dir, f'{id}.npy')
            data = np.load(file, allow_pickle=True).item()
            np_position = data['position']
            np_rotation = data['rotation']
            np_position = np_position.reshape(nframes, -1)
            np_rotation = aa2repr6d(torch.from_numpy(np_rotation)).numpy().reshape(nframes, -1)
            np_velocity = np.concatenate([np_position[1:] - np_position[:-1], np.zeros((1, 3*njoints))], axis=0)
        
        else:
            file = os.path.join(config.src_dir, f'{id}.npy')
            data = np.load(file, allow_pickle=True).item()
            np_position = data['position']
            np_rotation = data['rotation']
            np_position = np_position.reshape(nframes, -1)
            np_rotation = aa2repr6d(torch.from_numpy(np_rotation)).numpy().reshape(nframes, -1)
            np_velocity = np.concatenate([np_position[1:] - np_position[:-1], np.zeros((1, 3*njoints))], axis=0)
            
        # print(np_position.shape, np_rotation.shape, np_velocity.shape)
        motion = torch.from_numpy(np.concatenate([np_position, np_rotation, np_velocity], axis=-1)).to(config.device).float().unsqueeze(0)
        if shape is None and shapes is None:
            shape = motion.shape
            shapes = [np_position.shape, np_rotation.shape, np_velocity.shape]
        motion_latent = motion_extractor(motion)[1].detach().cpu().numpy().squeeze()
        np.save(os.path.join(save_path, f'{id}.npy'), motion_latent)
    
    print(f"{model_type} motion shape: ", shape)

if __name__ == '__main__':
    common.fixseed(1024)

    parser = argparse.ArgumentParser(description='### Extracting ###')
    parser.add_argument('-i', '--test_data', default='data/sam_test', type=str)
    parser.add_argument('--src_dir', default='./eval/pred/npy', type=str)
    parser.add_argument('--feat_save', default='./all_features', type=str)
    parser.add_argument('--model', default='mospa', type=str)
    args = parser.parse_args()
    config = Namespace()
    config.src_dir = args.src_dir
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.extractor = './save/extractor/weights_1500.pt'
    config.test_data = args.test_data
    config.feat_save = args.feat_save
    config.model = args.model
    config.latent_dim = 1024

    audio_dim, motion_dim = 2276, 300
    if config.model == 'mospa' or config.model == 'mospa_v2' or config.model == 'gt':
        extract_audio_features(config, audio_dim, latent_dim=config.latent_dim)
        extract_gt_motion_features(config, motion_dim, latent_dim=config.latent_dim)
    extract_motion_features(config, config.model, motion_dim, latent_dim=config.latent_dim)