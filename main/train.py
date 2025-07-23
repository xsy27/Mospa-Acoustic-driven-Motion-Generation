import os
import time
import torch
import shutil
import argparse
import utils.common as common

from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from utils.logger import Logger
from network.models import MotionDiffusion
from network.training import MotionTrainingPortal

from diffusion.create_diffusion import create_gaussian_diffusion
from config.option import add_model_args, add_data_args, add_train_args, add_diffusion_args, config_parse

from data.dataloader import prepare_train_dataloader, prepare_val_dataloader

def train(config, resume, logger, tb_writer):
    
    common.fixseed(1024)
    np_dtype = common.select_platform(32)
    
    print("Loading dataset..")

    train_dataloader, vec_len, audio_dim = prepare_train_dataloader(config, dtype=np_dtype)
    val_dataloader = prepare_val_dataloader(config, dtype=np_dtype)
    print(f'vec_len: {vec_len} | audio_dim: {audio_dim}')
    
    diffusion = create_gaussian_diffusion(config)

    model = MotionDiffusion(config.dataset.pose_vec, vec_len, audio_dim, config.dataset.clip_len, config.dataset.binaural,
                config.arch.latent_dim, config.arch.ff_size, 
                config.arch.num_layers, config.arch.num_heads, 
                arch=config.arch.decoder, cond_mask_prob=config.trainer.cond_mask_prob, mask_sound_source=config.trainer.mask_sound_source,
                mask_genre=config.trainer.mask_genre, device=config.device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(config.device)
    
    # logger.info('\nModel structure: \n%s' % str(model))
    trainer = MotionTrainingPortal(config, model, diffusion, train_dataloader, val_dataloader, logger, tb_writer)
    
    if resume is not None:
        try:
            trainer.load_checkpoint(resume)
        except FileNotFoundError:
            print('No checkpoint found at %s' % resume); exit()
    
    trainer.run_loop()

if __name__ == '__main__':
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='### Generative Training ###')
    
    # Runtime parameters
    parser.add_argument('-n', '--name', default='debug', type=str, help='The name of this training')
    parser.add_argument('-c', '--config', default='./config/mospa.json', type=str, help='config file path (default: None)')
    parser.add_argument('-i', '--data', default='data/sam_train', type=str)
    parser.add_argument('-v', '--val_data', default='data/sam_val', type=str)
    parser.add_argument('-t', '--test_data', default='data/sam_test', type=str)
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('-s', '--save', default='./save', type=str, help='show the debug information')
    parser.add_argument('--smpl_dir', default='smpl_models/smplx', type=str)
    parser.add_argument('--finetune', action='store_true')
    add_model_args(parser); add_data_args(parser); add_diffusion_args(parser); add_train_args(parser)
    
    args = parser.parse_args()
    
    if args.config:
        config = config_parse(args)
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if 'debug' in args.name:
        config.arch.offset_frame = config.arch.clip_len
        config.trainer.workers = 1
        config.trainer.load_num = -1
        config.trainer.batch_size = 256

    if os.path.exists(config.save) and 'debug' not in args.name and args.resume is None:
        proceed = input("Warning! Save directory already exists! Do you want to remove previous and proceed? (Y/N): ").lower()
        if proceed == "y":
            shutil.rmtree(config.save, ignore_errors=True)
        elif proceed == "n":
            quit()
        else:
            print("Invalid input!")
            quit()
    
    os.makedirs(config.save, exist_ok=True)

    logger = Logger('%s/training.log' % config.save)
    tb_writer = SummaryWriter(log_dir='%s/runtime' % config.save)
    
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open('%s/config.json' % config.save, 'w') as f:
        f.write(str(config))
    f.close()
    
    logger.info('\Generative locamotion training with config: \n%s' % config)
    train(config, args.resume, logger, tb_writer)
    logger.info('\nTotal training time: %s mins' % ((time.time() - start_time) / 60))