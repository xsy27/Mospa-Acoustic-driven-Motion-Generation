import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm
from utils.visualizer import export
from utils.nn_transforms import repr6d2aa
from torch.utils.data.distributed import DistributedSampler
import random

NJOINTS = 25

class AmoTrain(Dataset):
    def __init__(self, audios, motions, interval, names, betas, ssl):
        self.audios = audios
        self.motions = motions
        self.mask = np.ones(interval, dtype=bool) if interval is not None else None
        self.names = names
        self.betas = betas
        self.ssl = ssl
        assert(len(self.audios) == len(self.motions))
        assert(len(self.names) == len(self.motions))

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, index):
        return {
            'data': self.motions[index],
            'conditions': {
                'audio': self.audios[index],
                'state': int(self.names[index].__str__()[-1]),
            },
            'ssl': {
                'ssl': self.ssl[index],
            },
            'configs': {
                'betas': self.betas[index],
                'mask': self.mask
            }
        }
    
class AMoTest(Dataset):
    def __init__(self, audios, motions, names, masks, betas, ssl):
        self.audios = audios
        self.motions = motions
        self.names = names
        self.masks = masks
        self.betas = betas
        self.ssl = ssl
        assert(len(self.audios) == len(self.motions))
        assert(len(self.names) == len(self.motions))
        assert(len(self.masks) == len(self.motions))

    def __len__(self):
        return len(self.motions)
    
    def __getitem__(self, index):
        return {
            'data': self.motions[index],
            'conditions': {
                'audio': self.audios[index],
                'state': int(self.names[index].__str__()[-1]),
            },
            'ssl': {
                'ssl': self.ssl[index],
            },
            'configs': {
                'betas': self.betas[index],
                'name': self.names[index].__str__(),
                'mask': self.masks[index]
            }
        }

def load_train_data(data_dir, dtype=np.float32, interval=120, move=30, name='SAM', ssl=False):
    tot = 0
    audio_data, motion_data, smpl_batas_data, input_names = [], [], [], []
    ssl_data= []
    fnames = sorted(os.listdir(data_dir))

    print("Loading train data...")
    for fname in tqdm(fnames):
        path = os.path.join(data_dir, fname)
        with open(path, 'rb') as f:
            data = np.load(f)
            id = data['id']
            np_audio = data['audio_array']
            np_motion = data['motion_array']
            np_ssl = data['ssl_array']
            smpl_betas = data['smpl_betas']
        audio_sample_rate = 1
        seq_len, dim = np_audio.shape
        for i in range(0, seq_len, move):
            i_sample = i // audio_sample_rate
            interval_sample = interval // audio_sample_rate

            audio_sub_seq = np_audio[i_sample: i_sample + interval_sample]
            motion_sub_seq = np_motion[i: i + interval]
            ssl_sub_seq = np_ssl[i: i + interval]

            if len(audio_sub_seq) == interval_sample and len(motion_sub_seq) == interval:
                audio_data.append(audio_sub_seq)
                motion_data.append(motion_sub_seq)
                ssl_data.append(ssl_sub_seq)
                smpl_batas_data.append(smpl_betas)
                input_names.append(id)

                # tot += 1
                # if tot > 1:
                #     break

        # tot += 1
        # if tot > 1:
        #     break
        
        # tot += 1
        # if tot > 9:
        #     break

    return audio_data, motion_data, input_names, smpl_batas_data, ssl_data

def load_val_test_data(data_dir, save_dir, interval, dtype=np.float32, name='SAM',
                        ssl=False, save_gt=False, save_small_gt=False, test=False):
    tot = 0
    input_names = []

    audio_data, motion_data, masks, smpl_betas_data = [], [], [], []
    ssl_data= []
    sound_sources = []
    # idxs = random.choices(range(256), k=10)
    fnames = sorted(os.listdir(data_dir))
    if not test: fnames = random.choices(fnames, k=256)
    # fnames = fnames[:256]
    # fnames = [fnames[i] for i in idxs]
    
    print("Loading validation data...")
    for fname in tqdm(fnames):
        path = os.path.join(data_dir, fname)
        with open(path, 'rb') as f:
            data = np.load(f)
            id = data['id'].__str__()
            np_audio = data['audio_array']
            np_motion = data['motion_array']
            np_ssl = data['ssl_array']
            smpl_betas = data['smpl_betas']

        audio_data.append(np_audio[:interval])
        motion_data.append(np_motion[:interval])
        ssl_data.append(np_ssl[:interval])
        smpl_betas_data.append(smpl_betas)
        masks.append(np.ones(interval, dtype=bool))
        input_names.append(id)

        sound_sources.append(np_ssl[:interval])

        # tot += 1
        # if tot > 9:
        #     break

    if not ssl:
        export_gt(motion_data, input_names, sound_sources, '%s/%s/' % (save_dir, 'gt'))

    return audio_data, motion_data, input_names, masks, smpl_betas_data, ssl_data

def prepare_train_dataloader(config, dtype=np.float32, ssl=False):
    train_audio_data, train_motion_data, train_names, betas, train_ssl= load_train_data(
        config.data, dtype=dtype, interval=config.dataset.clip_len, move=config.dataset.move, name=config.dataset.dataset_name, ssl=ssl)
    vec_len, audio_dim = train_motion_data[0].shape[-1], train_audio_data[0].shape[-1]
    data = AmoTrain(train_audio_data, train_motion_data, config.dataset.clip_len, train_names, betas, train_ssl)
    sampler = RandomSampler(data, replacement=True)
    data_loader = DataLoader(
        data,
        num_workers=config.trainer.workers,
        batch_size=config.trainer.batch_size,
        sampler=sampler,
        pin_memory=True
    )
    return data_loader, vec_len, audio_dim

def prepare_feature_train_dataloader(config, dtype=np.float32, ssl=False):
    train_audio_data, train_motion_data, train_names, train_betas, train_ssl = load_train_data(
        "data/sam_train", dtype=dtype, interval=config.dataset.clip_len, move=config.dataset.move, name=config.dataset.dataset_name, ssl=ssl)
    val_audio_data, val_motion_data, val_names, val_betas, val_ssl = load_train_data(
        "data/sam_val", dtype=dtype, interval=config.dataset.clip_len, move=config.dataset.move, name=config.dataset.dataset_name, ssl=ssl)
    test_audio_data, test_motion_data, test_names, test_betas, test_ssl = load_train_data(
        "data/sam_test", dtype=dtype, interval=config.dataset.clip_len, move=config.dataset.move, name=config.dataset.dataset_name, ssl=ssl)
    vec_len, audio_dim = train_motion_data[0].shape[-1], train_audio_data[0].shape[-1]
    train_audio_data, train_motion_data, train_names, train_betas, train_ssl = \
    np.concatenate([train_audio_data, val_audio_data, test_audio_data], axis=0), \
    np.concatenate([train_motion_data, val_motion_data, test_motion_data], axis=0), \
    np.concatenate([train_names, val_names, test_names], axis=0), \
    np.concatenate([train_betas, val_betas, test_betas], axis=0), \
    np.concatenate([train_ssl, val_ssl, test_ssl], axis=0)
    data = AmoTrain(train_audio_data, train_motion_data, config.dataset.clip_len, train_names, train_betas, train_ssl)
    sampler = RandomSampler(data, replacement=True)
    data_loader = DataLoader(
        data,
        num_workers=config.trainer.workers,
        batch_size=config.trainer.batch_size,
        sampler=sampler,
        pin_memory=True
    )
    return data_loader, vec_len, audio_dim

def prepare_val_dataloader(config, dtype=np.float32, ssl=False):
    val_audio_data, val_motion_data, val_names, val_masks, betas, val_ssl = load_val_test_data(
        config.val_data, config.save, config.dataset.clip_len, dtype=dtype, name=config.dataset.dataset_name, 
        ssl=ssl)
    data = AMoTest(val_audio_data, val_motion_data, val_names, val_masks, betas, val_ssl)
    data_loader = DataLoader(
        data,
        num_workers=config.trainer.workers,
        batch_size=128,
        shuffle=False
    )
    return data_loader

def prepare_test_dataloader(config, dtype=np.float32, ssl=False, save_gt=False):
    test_audio_data, test_motion_data, test_names, test_masks, betas, test_ssl = load_val_test_data(
        config.test_data, config.save, config.dataset.clip_len, dtype=dtype, name=config.dataset.dataset_name, 
        ssl=ssl, test=True, save_gt=save_gt)
    vec_len, audio_dim = test_motion_data[0].shape[-1], test_audio_data[0].shape[-1]
    data = AMoTest(test_audio_data, test_motion_data, test_names, test_masks, betas, test_ssl)
    data_loader = DataLoader(
        data,
        num_workers=config.workers,
        batch_size=1,
        shuffle=False
    )
    return data_loader, vec_len, audio_dim

def export_gt(motions, input_names, sound_sources, save_dir):
    x_start_poss = []
    x_start_rots = []
    for motion in motions:
        nframes = motion.shape[0]
        x_start_pos = motion[..., 0:3*NJOINTS].reshape(nframes, NJOINTS, 3)
        x_start_poss.append(x_start_pos)

        motion = torch.from_numpy(motion)
        x_start_rot = repr6d2aa(motion[..., 3*NJOINTS:9*NJOINTS].view(nframes, NJOINTS, 6)).float()
        x_start_rot = x_start_rot.detach().cpu().numpy()
        x_start_rots.append(x_start_rot)

    export(x_start_poss, x_start_rots, sound_sources, input_names, save_dir, prefix='gt')