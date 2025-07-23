import copy
import json
from types import SimpleNamespace as Namespace

def add_model_args(parser):
    parser.add_argument('--decoder', type=str, help='Type of decoder to use.')
    parser.add_argument('--latent_dim', type=int, help='Width of the Transformer/GRU layers.')
    parser.add_argument('--ff_size', type=int, help='Feed-forward size for Transformer/GRU.')
    parser.add_argument('--num_heads', type=int, help='Number of attention heads in the Transformer.')
    parser.add_argument('--num_layers', type=int, help='Number of layers in the model.')


def add_data_args(parser):
    parser.add_argument('--clip_len', type=int, help='Clip length of the audio and the motion')
    parser.add_argument('--move', type=int, help='Move of the clip')
    parser.add_argument('--rot_rep', type=str, help="Rotation representation")
    parser.add_argument('--cond', default=None, type=str, help='Condition')
    parser.add_argument('--dataset_name', type=str, help='Dataset')
    parser.add_argument('--binaural', type=bool, help='If the audio is binaural')


def add_diffusion_args(parser):
    parser.add_argument('--noise_schedule', type=str, help='Noise schedule: choose from "cosine", "linear", or "linear1".')
    parser.add_argument('--diffusion_steps', type=int, help='Number of diffusion steps.')
    parser.add_argument('--sigma_small', type=bool, help='Use small sigma values.')


def add_train_args(parser):
    parser.add_argument('--epoch', type=int, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, help='Learning rate for training.')
    parser.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of steps to anneal the learning rate.")
    parser.add_argument('--weight_decay', default=0.00, type=float, help='Weight decay for the optimizer.')
    parser.add_argument('--batch_size', type=int, help='Batch size for training.')
    parser.add_argument('--loss_terms', type=str, help='Loss terms to use in training. Format: [mse_rotation, positional_loss, velocity_loss, foot_contact]. Use 0 for No, 1 for Yes, e.g., "1111".')
    parser.add_argument('--cond_mask_prob', type=float, help='Probability of masking conditioning.')
    parser.add_argument('--ema', default=False, type=bool, help='Use Exponential Moving Average (EMA) for model parameters.')
    parser.add_argument('--workers', type=int, help='Number of workers for data loading.')


def config_parse(args):
    config = copy.deepcopy(json.load(open(args.config), object_hook=lambda d: Namespace(**d)))

    config.data = args.data
    config.val_data = args.val_data
    config.test_data = args.test_data
    config.name = args.name
    config.finetune = args.finetune

    config.arch.decoder = str(args.decoder) if args.decoder is not None else config.arch.decoder
    config.arch.latent_dim = int(args.latent_dim) if args.latent_dim is not None else config.arch.latent_dim
    config.arch.ff_size = int(args.ff_size) if args.ff_size is not None else config.arch.ff_size
    config.arch.num_heads = int(args.num_heads) if args.num_heads is not None else config.arch.num_heads
    config.arch.num_layers = int(args.num_layers) if args.num_layers is not None else config.arch.num_layers

    config.dataset.pose_vec = config.dataset.pose_vec
    config.dataset.clip_len = int(args.clip_len) if args.clip_len is not None else config.dataset.clip_len
    config.dataset.move = int(args.move) if args.move is not None else config.dataset.move
    config.dataset.rot_rep = str(args.rot_rep) if args.rot_rep is not None else config.dataset.rot_rep
    config.dataset.njoints = config.dataset.njoints
    config.dataset.nfeats = config.dataset.nfeats
    config.dataset.cond = str(args.cond) if args.cond is not None else config.dataset.cond
    config.dataset.smpl_dir = str(args.smpl_dir) if args.smpl_dir is not None else config.dataset.smpl_dir
    config.dataset.dataset_name = str(args.dataset_name) if args.dataset_name is not None else config.dataset.dataset_name
    config.dataset.binaural = bool(args.binaural) if args.binaural is not None else config.dataset.binaural
        
    config.diff.noise_schedule = str(args.noise_schedule) if args.noise_schedule is not None else config.diff.noise_schedule
    config.diff.diffusion_steps = int(args.diffusion_steps) if args.diffusion_steps is not None else config.diff.diffusion_steps
    config.diff.sigma_small = True if args.sigma_small else config.diff.sigma_small

    config.trainer.epoch = int(args.epoch) if args.epoch is not None else config.trainer.epoch
    config.trainer.lr = float(args.lr) if args.lr is not None else config.trainer.lr
    config.trainer.weight_decay = args.weight_decay
    config.trainer.lr_anneal_steps = args.lr_anneal_steps
    config.trainer.cond_mask_prob = args.cond_mask_prob if args.cond_mask_prob is not None else config.trainer.cond_mask_prob
    config.trainer.mask_sound_source = config.trainer.mask_sound_source
    config.trainer.mask_genre = config.trainer.mask_genre
    config.trainer.batch_size = int(args.batch_size) if args.batch_size is not None else config.trainer.batch_size
    config.trainer.ema = True #if args.ema else config.trainer.ema
    config.trainer.workers = int(args.workers) if args.workers is not None else config.trainer.workers
    loss_terms = args.loss_terms if args.loss_terms is not None else config.trainer.loss_terms
    config.trainer.use_loss_mse = True if loss_terms[0] == '1' else False
    config.trainer.use_loss_traj = True if loss_terms[1] == '1' else False
    config.trainer.use_loss_rotation = True if loss_terms[2] == '1' else False
    config.trainer.use_loss_geo = True if loss_terms[3] == '1' else False
    config.trainer.use_loss_foot_contact = True if loss_terms[4] == '1' else False
    config.trainer.load_num = -1
    # config.trainer.save_freq = int(config.trainer.epoch // 10)
    config.trainer.save_freq = int(config.trainer.epoch // 5) if config.trainer.epoch <= 1000 else 500
    config.trainer.eval_freq = int(config.trainer.epoch // 5) if config.trainer.epoch <= 1000 else 500

    data_prefix = args.data.split('/')[-1].split('.')[0]
    config.save = '%s/%s' % (args.save, args.name)
    return config

# evaluation
def add_eval_args(parser):
    parser.add_argument('--workers', type=int, help='Number of workers for data loading.')
    parser.add_argument('--diffusion_steps', type=int, help='Number of diffusion steps.')

def eval_config_parse(args):
    config = copy.deepcopy(json.load(open(args.config), object_hook=lambda d: Namespace(**d)))
    config.test_data = args.test_data
    config.smpl_dir = args.smpl_dir
    config.ckpt = args.ckpt
    config.save = '%s/%s' % (args.save, args.model)
    config.workers = args.workers if args.workers is not None else 8
    config.save_pred = args.save_pred
    config.save_small_pred = args.save_small_pred
    config.feat_save = args.feat_save
    config.model = args.model
    config.metrics = args.metrics

    config.diff.diffusion_steps = args.diffusion_steps if args.diffusion_steps is not None else config.diff.diffusion_steps

    return config