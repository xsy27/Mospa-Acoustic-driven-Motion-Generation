import os
import argparse
from argparse import Namespace
import torch
from tqdm import tqdm
from network.models import MotionDiffusion
import numpy as np
import blobfile as bf
from utils.nn_transforms import repr6d2aa
from smplx import SMPLX
from utils.visualizer import export
import shutil
from data.dataloader import prepare_test_dataloader
from diffusion.create_diffusion import create_gaussian_diffusion
from utils.metrics import calc_metrics, calc_metrics_simple
from config.option import add_eval_args, eval_config_parse
from torch.nn import functional as F
from utils.smplx import smplx_forward, smplx_FK, get_smplx_joints
import random

NJOINTS = 25

def predict(config, dataloader, model, diffusion, smpl, ssl=None):
    gt_xyz = []
    pred_xyz = []
    pred_rot = []
    sound_sources = []
    fnames = []
    skeletons, parents = get_smplx_joints(model_path=config.dataset.smpl_dir, gender="NEUTRAL_2020", njoints=NJOINTS)
    skeletons = skeletons.to(config.device)
    batch_size, frame_num, vec_len = 1, 240, 300
    for datas in tqdm(dataloader):
        datas = {key: val.to(config.device) if torch.is_tensor(val) else val for key, val in datas.items()}
        cond = {key: val.to(config.device) if torch.is_tensor(val) else val for key, val in datas['conditions'].items()}
        ssl_features = {key: val.to(config.device) if torch.is_tensor(val) else val for key, val in datas['ssl'].items()}
        data_config = {key: val.to(config.device) if torch.is_tensor(val) else val for key, val in datas['configs'].items()}
        x_start = datas['data'].float()

        # features
        assert ssl_features['ssl'].shape[-1] == 3
        pred_ss = F.normalize(ssl_features['ssl'].float(), dim=-1)
        cond['pred_ss'] = pred_ss

        # permute
        cond['pred_ss'] = cond['pred_ss'].permute(0, 2, 1) # [bs, 3, frame_num]
    
        x_start = x_start.permute(0, 2, 1) # [bs, vec_len, frame_num]
        cond['audio'] = cond['audio'].permute(0, 2, 1).float() # [bs, audio_dim, frame_num]

        # predict
        assert vec_len == 12 * NJOINTS
        
        x_t = torch.randn_like(x_start)
        conditions = {'y': cond}
        data_shape = x_start.shape
        model_output = diffusion.p_sample_loop(model, data_shape, clip_denoised=False, model_kwargs=conditions, skip_timesteps=0,
                                                init_image=None, dump_steps=None, noise=None, const_noise=False)
        model_output = model_output.permute(0, 2, 1)
        x_start = x_start.permute(0, 2, 1)
        assert ((batch_size, frame_num, vec_len) == model_output.shape)

        model_output_rot = repr6d2aa(model_output[..., 3*NJOINTS:9*NJOINTS].view(batch_size, frame_num, NJOINTS, 6)).float() # [bs, frame_num, 24, 3]
        model_output_transl = model_output[..., 0:3].float()

        # model_output_rot = repr6d2aa(x_start[..., 3*NJOINTS:9*NJOINTS].view(bs, frame_num, NJOINTS, 6)).float() # [bs, frame_num, 24, 3]
        # model_output_transl = x_start[..., 0:3].float()

        # save
        smpl_poses = model_output_rot.view(batch_size, frame_num, -1)
        pred_fwd = smplx_FK(skeletons, parents, batch_size, model_output_rot, model_output_transl)
        for i in range(batch_size):
            gt_xyz.append(x_start[i, :, :3*NJOINTS].cpu().detach().numpy().reshape(frame_num, NJOINTS, 3))
            pred_xyz.append(pred_fwd[i].cpu().detach().numpy().reshape(frame_num, NJOINTS, 3))
            pred_rot.append(smpl_poses[i].cpu().detach().numpy().reshape(frame_num, NJOINTS, 3))
            fnames.append(data_config['name'][i])
            assert ssl_features['ssl'].shape == (1, frame_num, 3)
            sound_sources.append(ssl_features['ssl'][i].cpu().numpy().reshape(frame_num, 3))
    
    if(config.save_small_pred):
        vis = random.choices(range(len(pred_xyz)), k=10)
        pred_xyz_small, pred_rot_small, sound_sources_small, fnames_small = [pred_xyz[i] for i in vis], [pred_rot[i] for i in vis], [sound_sources[i] for i in vis], [fnames[i] for i in vis]
        export(pred_xyz_small, pred_rot_small, sound_sources_small, fnames_small, '%s/%s/' % (config.save, 'pred'), prefix='pred')
    
    if(config.save_pred and not config.save_small_pred):
        export(pred_xyz, pred_rot, sound_sources, fnames, '%s/%s/' % (config.save, 'pred'), prefix='pred')
    
    return np.array(gt_xyz), np.array(pred_xyz) # [nsamples, nframes, njoints, 3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', default='./config/mospa.json', type=str, help='config file path (default: None)')
    parser.add_argument('--test_data', type=str, default='data/sam_test', help='input local dictionary for SAM evaluation dataset.')
    parser.add_argument('--smpl_dir', default='smpl_models/smplx', type=str)
    parser.add_argument('--ckpt', type=str, default='save/mospa_sam_train/weights_10000.pt', help='input local dictionary that stores the best model parameters.')
    parser.add_argument('--save', type=str, default='eval', help='input local dictionary that stores the output data.')
    parser.add_argument('--save_pred', type=bool, default=False)
    parser.add_argument('--save_small_pred', type=bool, default=False)
    parser.add_argument('--feat_save', default='./all_features', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--metrics', action='store_true')
    add_eval_args(parser)

    args = parser.parse_args()
    config = eval_config_parse(args)

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.extractor = './save/extractor/weights_1000.pt'
    config.src_dir = '%s/%s/%s' % (config.save, "pred", "npy")
    config.feat_dir = './all_features'

    if os.path.exists(config.save):
        proceed = input("Warning! Save directory already exists! Do you want to remove previous and proceed? (Y/N): ").lower()
        if proceed == "y":
            shutil.rmtree(config.save, ignore_errors=True)
        elif proceed == "n":
            quit()
        else:
            print("Invalid input!")
            quit()
        shutil.rmtree(config.save, ignore_errors=True)
    os.makedirs(config.save, exist_ok=True)

    # test_data_loader, vec_len, audio_dim = prepare_test_dataloader(config)
    test_data_loader, vec_len, audio_dim = prepare_test_dataloader(config)

    model = MotionDiffusion(config.dataset.pose_vec, vec_len, audio_dim, config.dataset.clip_len, config.dataset.binaural,
                config.arch.latent_dim, config.arch.ff_size, 
                config.arch.num_layers, config.arch.num_heads, 
                arch=config.arch.decoder, mask_sound_source=config.trainer.mask_sound_source,
                mask_genre=config.trainer.mask_genre, device=config.device).to(config.device)
    
    smpl = SMPLX(model_path=config.dataset.smpl_dir,
                    use_face_contour=False,
                    gender='NEUTRAL_2020', 
                    num_betas=300,
                    num_expression_coeffs=100,
                    use_pca=False,
                    batch_size=config.dataset.clip_len).eval().to(config.device)
    
    ssl_model = None
    
    if bf.exists(config.ckpt):
        ckpt_model = torch.load(config.ckpt)
        model.load_state_dict(ckpt_model['state_dict'])
    else:
        raise ValueError("loadpoint does not exist!!!")
    model.eval()

    diffusion = create_gaussian_diffusion(config)
    
    gt_xyz, pred_xyz = predict(config, test_data_loader, model, diffusion, smpl, ssl=ssl_model)

    result_path = os.path.join(config.save, "metrics.txt")

    if config.metrics: 
        result = calc_metrics(config, gt_xyz, pred_xyz)
    else: 
        result = calc_metrics_simple(gt_xyz, pred_xyz)

    with open(result_path, 'a') as f:
        f.write(str(result))
        f.write('\n')

    print(result)