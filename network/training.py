import numpy as np
import blobfile as bf
import utils.common as common
from tqdm import tqdm
from smplx import SMPLX

import torch
from torch.optim import AdamW
from torch.utils.data import Subset, DataLoader
from torch_ema import ExponentialMovingAverage
from torch.nn import functional as F

from diffusion.resample import create_named_schedule_sampler
from diffusion.gaussian_diffusion import *

from utils.nn_transforms import repr6d2aa
from utils.smplx import smplx_forward, smplx_FK, get_smplx_joints

class BaseTrainingPortal:
    def __init__(self, config, model, diffusion, train_dataloader, val_dataloader, logger, tb_writer, prior_loader=None):
        
        self.model = model
        self.diffusion = diffusion
        self.dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger
        self.tb_writer = tb_writer
        self.config = config
        self.batch_size = config.trainer.batch_size
        self.lr = config.trainer.lr
        self.lr_anneal_steps = config.trainer.lr_anneal_steps

        self.epoch = 0
        self.num_epochs = config.trainer.epoch
        self.save_freq = config.trainer.save_freq
        self.best_loss = 1e10
        
        print('Train with %d epoches, %d batches by %d batch_size' % (self.num_epochs, len(self.dataloader), self.batch_size))

        self.save_dir = config.save

        self.smplx = SMPLX(model_path=self.config.dataset.smpl_dir,
                    use_face_contour=False,
                    gender='NEUTRAL_2020', 
                    num_betas=300,
                    num_expression_coeffs=100,
                    use_pca=False,
                    batch_size=config.dataset.clip_len).eval().to(config.device)
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=config.trainer.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.num_epochs, eta_min=self.lr * 0.1)
        
        if config.trainer.ema:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        
        self.device = config.device

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.use_ddp = False
        
        self.prior_loader = prior_loader

        self.pose_vec = config.dataset.pose_vec
        self.njoints = config.dataset.njoints

        self.skeletons, self.parents = get_smplx_joints(model_path=self.config.dataset.smpl_dir, gender='NEUTRAL_2020', njoints=self.njoints)
        self.skeletons = self.skeletons.to(self.device)

        self.finetune = config.finetune
        
    def diffuse(self, x_start, t, cond, data_config, noise=None, return_loss=False):
        raise NotImplementedError('diffuse function must be implemented')
    
    def validate(self, dataloader, save_folder_name):
        raise NotImplementedError('validate function must be implemented')
        
    def run_loop(self):
        self.validate(self.val_dataloader, save_folder_name='init_validation')
        
        epoch_process_bar = tqdm(range(self.epoch, self.num_epochs), desc=f'Epoch {self.epoch}')
        for epoch_idx in epoch_process_bar:
            self.model.train()
            self.model.training = True
            self.epoch = epoch_idx + 1
            if self.epoch >= 5000: self.finetune = True
            epoch_losses = {}
            
            for datas in self.dataloader:
                datas = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas.items()}
                cond = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['conditions'].items()}
                ssl_features = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['ssl'].items()}
                data_config = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['configs'].items()}
                x_start = datas['data'].float()

                self.opt.zero_grad()
                t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)
                
                _, losses = self.diffuse(x_start, t, cond, ssl_features, data_config, noise=None, return_loss=True)
                total_loss = (losses["loss"] * weights).mean()
                total_loss.backward()
                self.opt.step()
            
                if self.config.trainer.ema:
                    self.ema.update()
                
                for key_name in losses.keys():
                    if 'loss' in key_name:
                        if key_name not in epoch_losses.keys():
                            epoch_losses[key_name] = []
                        epoch_losses[key_name].append(losses[key_name].mean().item())
            
            loss_str = ''
            for key in epoch_losses.keys():
                loss_str += f'{key}: {np.mean(epoch_losses[key]):.6f}, '
            
            epoch_avg_loss = np.mean(epoch_losses['loss'])
            
            if self.epoch > 10 and epoch_avg_loss < self.best_loss: 
                self.save_checkpoint(filename='best')
            
            if epoch_avg_loss < self.best_loss:
                self.best_loss = epoch_avg_loss
            
            epoch_process_bar.set_description(f'Epoch {self.epoch}/{self.config.trainer.epoch} | loss: {epoch_avg_loss:.6f} | best_loss: {self.best_loss:.6f}')
            self.logger.info(f'Epoch {self.epoch}/{self.config.trainer.epoch} | {loss_str} | best_loss: {self.best_loss:.6f}')

            if self.epoch > 0  and self.epoch % 100 == 0:
                self.validate(self.val_dataloader, save_folder_name=f'validation_{self.epoch}')

            if self.epoch > 0 and self.epoch % self.config.trainer.save_freq == 0:
                self.save_checkpoint(filename=f'weights_{self.epoch}')
            
            for key_name in epoch_losses.keys():
                if 'loss' in key_name:
                    self.tb_writer.add_scalar(f'train/{key_name}', np.mean(epoch_losses[key_name]), self.epoch)

            self.scheduler.step()
        
        best_path = '%s/best.pt' % (self.config.save)
        self.load_checkpoint(best_path)
        # self.validate(self.val_dataloader, save_folder_name='best')

    def state_dict(self):
        model_state = self.model.state_dict()
        opt_state = self.opt.state_dict()
            
        return {
            'epoch': self.epoch,
            'state_dict': model_state,
            'opt_state_dict': opt_state,
            'config': self.config,
            'loss': self.best_loss,
        }

    def save_checkpoint(self, filename='weights'):
        save_path = '%s/%s.pt' % (self.config.save, filename)
        with bf.BlobFile(bf.join(save_path), "wb") as f:
            torch.save(self.state_dict(), f)
        self.logger.info(f'Saved checkpoint: {save_path}')

    def load_checkpoint(self, resume_checkpoint, load_hyper=True):
        if bf.exists(resume_checkpoint):
            checkpoint = torch.load(resume_checkpoint)
            self.model.load_state_dict(checkpoint['state_dict'])
            if load_hyper:
                self.epoch = checkpoint['epoch']
                self.best_loss = checkpoint['loss']
                self.opt.load_state_dict(checkpoint['opt_state_dict'])
            self.logger.info('\nLoad checkpoint from %s, start at epoch %d, loss: %.4f' % (resume_checkpoint, self.epoch, checkpoint['loss']))
        else:
            raise FileNotFoundError(f'No checkpoint found at {resume_checkpoint}')


class MotionTrainingPortal(BaseTrainingPortal):
    def __init__(self, config, model, diffusion, train_dataloader, val_dataloader, logger, tb_writer, finetune_loader=None):
        super().__init__(config, model, diffusion, train_dataloader, val_dataloader, logger, tb_writer, finetune_loader)

    def prepro(self, x_start, ssl_features, cond):

        if self.config.dataset.binaural:
            batch_size, frame_num, vec_len = x_start.shape

            # features
            assert ssl_features['ssl'].shape[-1] == 3
            cond['pred_ss'] = ssl_features['ssl'].float()

            # permute
            cond['pred_ss'] = cond['pred_ss'].permute(0, 2, 1) # [bs, 3, frame_num]
        
        x_start = x_start.permute(0, 2, 1) # [bs, vec_len, frame_num]
        cond['audio'] = cond['audio'].permute(0, 2, 1).float() # [bs, feature_num, frame_num]
        
        return x_start, cond

    def diffuse(self, x_start, t, cond, ssl_features, data_config, noise=None, return_loss=False, validate=False):
        x_start, cond = self.prepro(x_start, ssl_features, cond)
        
        batch_size, vec_len, frame_num = x_start.shape
        
        if noise is None:
            noise = th.randn_like(x_start)
        
        # the diffusion process
        if not validate:
            x_t = self.diffusion.q_sample(x_start, t, noise=noise)
            model_output = self.model.forward(x_t, self.diffusion._scale_timesteps(t), y=cond)
        else:
            x_t = th.randn_like(x_start)
            conditions = {'y': cond}
            data_shape = x_start.shape
            model_output = self.diffusion.p_sample_loop(self.model, data_shape, clip_denoised=False, model_kwargs=conditions, skip_timesteps=0,
                                                        init_image=None, dump_steps=None, noise=None, const_noise=False)
        # model_output: the output of the diffusion process, the shape is batch_size, pose_vector_length, frame_number
        assert ((batch_size, vec_len, frame_num) == model_output.shape)
        
        # calculate loss
        if return_loss:
            loss_terms = {}
            
            target = {
                ModelMeanType.PREVIOUS_X: self.diffusion.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.diffusion.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            mask = data_config['mask'].view(batch_size, 1, 1, -1) # [batch_size, 1, 1, frame_num]
            
            if self.config.trainer.use_loss_mse:
                loss_terms['loss_data'] = self.diffusion.masked_l2(
                    target.reshape(batch_size, self.njoints, -1, frame_num), 
                    model_output.reshape(batch_size, self.njoints, -1, frame_num), 
                    mask)
                target_vel = target[..., 1:] - target[..., :-1]
                model_output_vel = model_output[..., 1:] - model_output[..., :-1]
                loss_terms['loss_data_delta'] = self.diffusion.masked_l2(
                    target_vel.reshape(batch_size, self.njoints, -1, frame_num-1), 
                    model_output_vel.reshape(batch_size, self.njoints, -1, frame_num-1), 
                    mask[..., 1:])
                
            if self.config.trainer.use_loss_traj:
                loss_terms['loss_traj'] = self.diffusion.masked_l2(
                    target[:, 0:3, :].reshape(batch_size, 1, 3, frame_num),
                    model_output[:, 0:3, :].reshape(batch_size, 1, 3, frame_num),
                    mask)
                target_traj_vel = target[:, 0:3, 1:] - target[:, 0:3, :-1]
                model_output_traj_vel = model_output[:, 0:3, 1:] - model_output[:, 0:3, :-1]
                loss_terms['loss_traj_delta'] = self.diffusion.masked_l2(
                    target_traj_vel.reshape(batch_size, 1, 3, frame_num-1), 
                    model_output_traj_vel.reshape(batch_size, 1, 3, frame_num-1), 
                    mask[..., 1:])
                
                # a, b = (model_output[:, 0:3, -1] - model_output[:, 0:3, 0]), (target[:, 0:3, -1] - target[:, 0:3, 0])
                # dot_product = (a * b).sum(dim=-1)
                # norm_a = torch.norm(a, p=2, dim=-1)
                # norm_b = torch.norm(b, p=2, dim=-1)
                # loss_terms['loss_dir'] = (1 - dot_product / (norm_a * norm_b + 1e-8)) ** 2
            
            if self.config.trainer.use_loss_rotation:
                loss_terms['loss_rot'] = self.diffusion.masked_l2(
                    target[:, 3*self.njoints:9*self.njoints, :].reshape(batch_size, self.njoints, -1, frame_num), 
                    model_output[:, 3*self.njoints:9*self.njoints, :].reshape(batch_size, self.njoints, -1, frame_num), 
                    mask)
                target_rot_delta = target[:, 3*self.njoints:9*self.njoints, 1:] - target[:, 3*self.njoints:9*self.njoints, :-1]
                model_output_rot_delta = model_output[:, 3*self.njoints:9*self.njoints, 1:] - model_output[:, 3*self.njoints:9*self.njoints, :-1]
                loss_terms['loss_rot_delta'] = self.diffusion.masked_l2(
                    model_output_rot_delta.reshape(batch_size, self.njoints, -1, frame_num-1),
                    target_rot_delta.reshape(batch_size, self.njoints, -1, frame_num-1),
                    mask[..., 1:])
            
            if self.config.trainer.use_loss_geo or self.config.trainer.use_loss_foot_contact:
                target = target.permute(0, 2, 1) # [batch_size, frame_num, vec_len_new]
                model_output = model_output.permute(0, 2, 1) # [batch_size, frame_num, vec_len_new]
                target_xyz = target[..., 0:3*self.njoints].reshape(batch_size, frame_num, self.njoints, 3)

                model_output_rot = repr6d2aa(model_output[..., 3*self.njoints:9*self.njoints].reshape(batch_size, frame_num, self.njoints, 6)).float() # [bs, nframes, njoints, 3]
                model_output_transl = model_output[..., 0:3].float() # [bs, nframes, 3]

                target_rot_aa = repr6d2aa(target[..., 3*self.njoints:9*self.njoints].reshape(batch_size, frame_num, self.njoints, 6)).float() # [bs, nframes, njoints, 3]
                target_transl = target[..., 0:3].float()
                
                # smpl_poses = model_output_rot.view(batch_size, frame_num, -1) # [bs, nframes, njoints*3]
                # smpl_transls = model_output_transl
                # smpl_betass = data_config['betas']
                # pred_xyz = smplx_forward(self.smplx, batch_size, frame_num, smpl_poses, smpl_transls, smpl_betass)
                # pred_xyz = torch.cat(pred_xyz, dim=0).to(target_xyz.device).reshape(batch_size, frame_num, self.njoints, -1)
                pred_xyz = smplx_FK(self.skeletons, self.parents, batch_size, model_output_rot, model_output_transl) # [bs, nframes, njoints, 3]
                target_xyz = smplx_FK(self.skeletons, self.parents, batch_size, target_rot_aa, target_transl) # [bs, nframes, njoints, 3]

                assert pred_xyz.shape == target_xyz.shape
                if self.config.trainer.use_loss_geo:
                    loss_terms["loss_geo_xyz"] = self.diffusion.masked_l2(target_xyz.permute(0, 2, 3, 1), pred_xyz.permute(0, 2, 3, 1), mask)

                    target_xyz_vel = target_xyz[:, 1:] - target_xyz[:, :-1]
                    pred_xyz_vel = pred_xyz[:, 1:] - pred_xyz[:, :-1]
                    loss_terms["loss_geo_xyz_vel"] = self.diffusion.masked_l2(target_xyz_vel.permute(0, 2, 3, 1), pred_xyz_vel.permute(0, 2, 3, 1), mask[..., 1:])
                
                if self.config.trainer.use_loss_foot_contact:
                    l_foot_idx, r_foot_idx = 10, 11
                    l_ankle_idx, r_ankle_idx = 7, 8
                    relevant_joints = [l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx]
                    target_xyz_reshape = target_xyz.permute(0, 2, 3, 1)
                    pred_xyz_reshape = pred_xyz.permute(0, 2, 3, 1)
                    gt_joint_xyz = target_xyz_reshape[:, relevant_joints, :, :]  # [BatchSize, 4, 3, Frames]
                    gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2)  # [BatchSize, 4, Frames]
                    fc_mask = torch.unsqueeze((gt_joint_vel <= 0.01), dim=2).repeat(1, 1, 3, 1)
                    pred_joint_xyz = pred_xyz_reshape[:, relevant_joints, :, :]  # [BatchSize, 4, 3, Frames]

                    # another foot contact loss
                    # fc_mask = (pred_joint_xyz[:, :, 1:2, :].abs() <= 0.385).repeat(1, 1, 3, 1)[..., :-1]s

                    pred_vel = pred_joint_xyz[:, :, :, 1:] - pred_joint_xyz[:, :, :, :-1]
                    pred_vel[~fc_mask] = 0
                    loss_terms["loss_foot_contact"] = self.diffusion.masked_l2(pred_vel,
                                                torch.zeros(pred_vel.shape, device=pred_vel.device),
                                                mask[:, :, :, 1:])
                    
                target = target.permute(0, 2, 1)
                model_output = model_output.permute(0, 2, 1)
            
            loss_terms["loss"] = loss_terms.get('loss_data', 0.) + \
                                loss_terms.get('loss_data_delta', 0.) + \
                                loss_terms.get('loss_geo_xyz', 0) + \
                                loss_terms.get('loss_geo_xyz_vel', 0) + \
                                loss_terms.get('loss_foot_contact', 0.) + \
                                (3.0 if self.finetune else 1.0) * loss_terms.get('loss_traj', 0.) + \
                                (3.0 if self.finetune else 1.0) * loss_terms.get('loss_traj_delta', 0.) + \
                                (3.0 if self.finetune else 1.0) * loss_terms.get('loss_rot', 0.) + \
                                (3.0 if self.finetune else 1.0) * loss_terms.get('loss_rot_delta', 0.)
            
            return model_output.permute(0, 2, 1), loss_terms
        
        return model_output.permute(0, 2, 1)

    def validate(self, dataloader, save_folder_name):
        self.logger.info("Evaluation...")
        self.model.eval()
        self.model.training = False
        with torch.no_grad():
            pred_xyz = []
            pred_rot = []
            fnames = []
            eval_losses = {}

            for datas in dataloader:
                datas = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas.items()}
                cond = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['conditions'].items()}
                ssl_features = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['ssl'].items()}
                data_config = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['configs'].items()}

                x_start = datas['data'] # [bs, nframes, vec_len]
                bs, nframes, vec_len = x_start.shape
                t, _ = self.schedule_sampler.sample(dataloader.batch_size, self.device)
                model_output, losses = self.diffuse(x_start, t, cond, ssl_features, data_config, noise=None, return_loss=True, validate=True)
                assert model_output.shape == x_start.shape

                if self.epoch % self.config.trainer.eval_freq == 0:
                    model_output_rot = repr6d2aa(model_output[..., 3*self.njoints:9*self.njoints].view(bs, nframes, self.njoints, 6)).float() # [bs, nframes, njoints, 3]
                    model_output_transl = model_output[..., 0:3].float()
                    
                    smpl_poses = model_output_rot.view(bs, nframes, -1)
                    # smpl_transls = model_output_transl
                    # smpl_betass = data_config['betas']
                    # pred_fwd = smplx_forward(self.smplx, bs, nframes, smpl_poses, smpl_transls, smpl_betass)
                    pred_fwd = smplx_FK(self.skeletons, self.parents, bs, model_output_rot, model_output_transl)
                    for i in range(bs):
                        pred_xyz.append(pred_fwd[i].cpu().detach().numpy().reshape(nframes, self.njoints, 3))
                        pred_rot.append(smpl_poses[i].cpu().detach().numpy().reshape(nframes, self.njoints, 3))
                        fnames.append(data_config['name'][i])
                
                for key_name in losses.keys():
                    if 'loss' in key_name:
                        if key_name not in eval_losses.keys():
                            eval_losses[key_name] = []
                        eval_losses[key_name].append(losses[key_name].mean().item())
                
            for key_name in eval_losses.keys():
                if 'loss' in key_name:
                    self.tb_writer.add_scalar(f'val/{key_name}', np.mean(eval_losses[key_name]), self.epoch)
            
            # if self.epoch % self.config.trainer.eval_freq == 0:
            #     common.mkdir('%s/%s' % (self.save_dir, save_folder_name))
                # export(pred_xyz, pred_rot, fnames, '%s/%s/' % (self.save_dir, save_folder_name), prefix='pred')
        