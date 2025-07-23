import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from types import SimpleNamespace as Namespace
import copy
import json
from tqdm import tqdm
from torch.optim import Adam, AdamW
import blobfile as bf
from torch.utils.tensorboard import SummaryWriter
import shutil
import warnings

from utils.logger import Logger
from utils import common
from utils.nn_transforms import repr6d2aa
from data.dataloader import prepare_feature_train_dataloader
from network.models import *

warnings.filterwarnings('ignore')

class ExtractorTrainingPortal():
    def __init__(self, config, train_dataloader, motion_dim, audio_dim, hidden_size=1024, num_layers=4):
        self.num_epochs = config.trainer.epoch
        self.epoch = 0
        self.freeze = 1000
        self.config = config

        self.logger = Logger('%s/training.log' % config.save)
        self.tb_writer = SummaryWriter(log_dir='%s/runtime' % config.save)

        self.train_dataloader = train_dataloader
        self.hidden_size = hidden_size
        self.audio_dim = audio_dim
        self.motion_dim = motion_dim
        self.audio_extractor = AudioExtractor(audio_dim+4, hidden_size, num_layers).to(config.device)
        self.motion_extractor = MotionExtractor(motion_dim, hidden_size, num_layers).to(config.device)
        self.optimizer = Adam(list(self.audio_extractor.parameters()) + list(self.motion_extractor.parameters()), lr=config.trainer.lr)

        self.best_loss = 1e+5
    
    def state_dict(self):
        audio_extractor_state = self.audio_extractor.state_dict()
        motion_extractor_state = self.motion_extractor.state_dict()
        opt_state = self.optimizer.state_dict()
            
        return {
            'epoch': self.epoch,
            'audio_extractor_state_dict': audio_extractor_state,
            'motion_extractor_state_dict': motion_extractor_state,
            'opt_state_dict': opt_state,
            'config': self.config,
            'loss': self.best_loss,
        }

    
    def save_checkpoint(self, filename='weights'):
        save_path = '%s/%s.pt' % (self.config.save, filename)
        with bf.BlobFile(bf.join(save_path), "wb") as f:
            torch.save(self.state_dict(), f)
        self.logger.info(f'Saved checkpoint: {save_path}')
    
    def run_loop(self):
        self.logger.info(f"Config: {config}")
        epoch_process_bar = tqdm(range(self.num_epochs), desc=f'Epoch {self.epoch}')
        for epoch_idx in epoch_process_bar:
            self.audio_extractor.train()
            self.motion_extractor.train()
            epoch_losses = {}
            self.epoch = epoch_idx + 1

            # Freeze Autoencoder
            if self.epoch == self.freeze:
                for param in self.motion_extractor.parameters():
                    param.requires_grad = False
                for param in self.motion_extractor.encoder.parameters():
                    param.requires_grad = True
                self.optimizer = torch.optim.Adam(
                    list(filter(lambda p: p.requires_grad, self.motion_extractor.parameters())) + list(self.audio_extractor.parameters()),
                    lr=self.optimizer.param_groups[0]['lr']  # Preserve current LR
                )

            # Freeze Audio Feature Extractor
            # if self.epoch == 700:
            #     for param in self.audio_extractor.parameters():
            #         param.requires_grad = False
            #     self.optimizer = torch.optim.Adam(
            #         filter(lambda p: p.requires_grad, self.motion_extractor.parameters()), 
            #         lr=self.optimizer.param_groups[0]['lr']
            #     )

            for datas in self.train_dataloader:
                datas = {key: val.to(config.device) if torch.is_tensor(val) else val for key, val in datas.items()}
                cond = {key: val.to(config.device) if torch.is_tensor(val) else val for key, val in datas['conditions'].items()}
                ssl_features = {key: val.to(config.device) if torch.is_tensor(val) else val for key, val in datas['ssl'].items()}
                x_start = datas['data'].float()

                bs, nframes, _ = x_start.shape
                assert cond['audio'].shape == (bs, nframes, audio_dim)

                genres = cond['state'].view(bs, 1, 1).repeat(1, nframes, 1)
                ssl = ssl_features['ssl'] / torch.norm(ssl_features['ssl'], dim=-1, keepdim=True)
                audio = torch.concatenate([cond['audio'], ssl, genres], dim=-1).float()
                motion = x_start

                self.optimizer.zero_grad()
                losses = self.forward(audio, motion)
                total_loss = losses["loss"].mean()
                total_loss.backward()
                self.optimizer.step()

                for key_name in losses.keys():
                    if 'loss' in key_name:
                        if key_name not in epoch_losses.keys():
                            epoch_losses[key_name] = []
                        epoch_losses[key_name].append(losses[key_name].mean().item())
            
            loss_str = ''
            for key in epoch_losses.keys():
                loss_str += f'{key}: {np.mean(epoch_losses[key]):.6f}, '
            
            epoch_avg_loss = np.mean(epoch_losses['loss'])
            
            if epoch_avg_loss < self.best_loss:   
                self.best_loss = epoch_avg_loss          
                self.save_checkpoint(filename='best')
            
            epoch_process_bar.set_description(f'Epoch {self.epoch}/{self.num_epochs} | loss: {epoch_avg_loss:.6f} | best_loss: {self.best_loss:.6f}')
            self.logger.info(f'Epoch {self.epoch}/{self.num_epochs} | {loss_str} | best_loss: {self.best_loss:.6f}')
               
            if self.epoch > 0 and self.epoch % 100 == 0:
                self.save_checkpoint(filename=f'weights_{self.epoch}')
            
            for key_name in epoch_losses.keys():
                if 'loss' in key_name:
                    self.tb_writer.add_scalar(f'train/{key_name}', np.mean(epoch_losses[key_name]), self.epoch)
    
    def forward(self, audio, motion):
        bs, nframes, _ = audio.shape
        assert audio.shape[:2] == motion.shape[:2]

        output_audio, audio_features = self.audio_extractor(audio)
        output_motion, motion_features = self.motion_extractor(motion)

        # Bi-GRU
        assert audio_features.shape == (bs, 2*self.hidden_size)
        assert motion_features.shape == (bs, 2*self.hidden_size)

        loss_terms = {}

        loss_terms["loss_motion"] = nn.MSELoss()(output_motion, motion)

        m = config.arch.margin
        dist_sq = (torch.sum(audio_features**2, dim=1, keepdim=True) + 
           torch.sum(motion_features**2, dim=1, keepdim=True).t() - 
           2 * torch.matmul(audio_features, motion_features.t()))
        dist_sq = torch.clamp(dist_sq, min=0)
        D = torch.sqrt(dist_sq)
        pos_mask = torch.eye(bs, device=audio_features.device, dtype=torch.bool)
        neg_mask = ~pos_mask
        pos_loss = dist_sq[pos_mask].sum()
        neg_loss = (torch.clamp(m - D[neg_mask], min=0) ** 2).sum()
        total_cta_loss = pos_loss + neg_loss
        loss_terms["loss_cta"] = total_cta_loss / (bs * bs)

        loss_terms["loss"] = loss_terms["loss_cta"] + \
            (1.0 if self.epoch < self.freeze else 0.0) * loss_terms.get("loss_motion", 0.)

        return loss_terms

if __name__ == '__main__':
    common.fixseed(1024)

    parser = argparse.ArgumentParser(description='### Training ###')
    parser.add_argument('-i', '--data', default='data/sam_train', type=str)
    parser.add_argument('-c', '--config', default='./config/extractor.json', type=str, help='config file path')
    parser.add_argument('--save', default='./save/extractor0', type=str, help='save features path')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    args = parser.parse_args()
    config = copy.deepcopy(json.load(open(args.config), object_hook=lambda d: Namespace(**d)))
    config.data = args.data
    config.save = args.save
    config.part = None
    config.trainer.batch_size = args.batch_size if args.batch_size is not None else config.trainer.batch_size
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if os.path.exists(config.save):
        shutil.rmtree(config.save)
    os.makedirs(config.save)

    train_dataloader, motion_dim, audio_dim = prepare_feature_train_dataloader(config, dtype=np.float32)

    portal = ExtractorTrainingPortal(config, train_dataloader, motion_dim, audio_dim, 
                                     hidden_size=config.arch.hidden_size, num_layers=config.arch.num_layers)
    portal.run_loop()