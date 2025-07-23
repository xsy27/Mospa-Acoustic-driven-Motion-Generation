import sys
sys.path.insert(0, '.')
import bpy
from argparse import ArgumentParser
import os
import numpy as np
import shutil
from tqdm import tqdm

import utils.blender.launch # noqa
from utils.blender.scene import setup_scene
from utils.blender.render_mesh import npy2obj
from utils.blender.floor import plot_floor
from utils.blender.camera import Camera
from utils.blender.sampler import get_frameidx
from utils.blender.tools import delete_objs, render_current_frame
from utils.blender.materials import body_material
from utils.blender.video import Video

# Orange
GEN_SMPL = body_material(0.658, 0.214, 0.0114)
# Green
GT_SMPL = body_material(0.035, 0.415, 0.122)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--npy", type=str, default=None, help="npy motion file")
    parser.add_argument("--dir", type=str, default=None, help="npy motion folder")
    parser.add_argument("--mode", type=str, default="video", help="render target: video, sequence, frame")
    parser.add_argument("--res", type=str, default="high")
    parser.add_argument("--denoising", type=bool, default=True)
    parser.add_argument("--oldrender", type=bool, default=True)
    parser.add_argument("--accelerator", type=str, default='gpu', help='accelerator device')
    parser.add_argument("--device", type=int, nargs='+', default=[0], help='gpu ids')
    parser.add_argument("--smpl", type=str, default='./smpl_models/smplx')
    parser.add_argument("--always_on_floor", action="store_true", help='put all the body on the floor (not recommended)')
    parser.add_argument("--gt", action='store_true', help='green for gt, otherwise orange')
    parser.add_argument("--fps", type=int, default=60, help="the frame rate of the rendered video")
    parser.add_argument("--num", type=int, default=6, help="the number of frames rendered in 'sequence' mode")
    parser.add_argument("--exact_frame", type=float, default=0.5, help="the frame id selected under 'frame' mode ([0, 1])")
    parser.add_argument("--save", type=str, default=None)
    cfg = parser.parse_args()
    return cfg

def render(path, mode, smpl, gt=False,
           exact_frame=None, num=8, always_on_floor=False, denoising=True, oldrender=True,
           res="high", accelerator='gpu', device=[0], fps=60, save=None):
    
    parsed_name = ''
    parsed_name = os.path.basename(path).replace('.npy', '')
    results_dir_name = parsed_name + '_frames'
    results_dir = os.path.join(os.path.dirname(os.path.dirname(path)), results_dir_name) if save is None else os.path.join(save, results_dir_name)
    # if os.path.exists(results_dir): shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    if mode == 'video':
        pass
        # if os.path.exists(results_dir.replace("_frames", ".mp4")) or os.path.exists(results_dir):
        #     print(f"Rendered or under rendering {path}")
        #     returns
        
    elif mode == 'sequence':
        img_name = os.path.basename(path).replace('.npy', '')
        img_path = os.path.join(os.path.dirname(os.path.dirname(path)), img_name) if save is None else os.path.join(save, img_name)
        img_path = img_path + (".gt" if gt else ".pred")
    
    elif mode == 'sep_sequence':
        img_name = os.path.basename(path).replace('.npy', '')
        img_path = os.path.join(os.path.dirname(os.path.dirname(path)), img_name) if save is None else os.path.join(save, img_name)
        save_sep_dir = img_path
        img_path = img_path + (".gt" if gt else ".pred") + '.png'
        os.makedirs(save_sep_dir, exist_ok=True)
    
    elif mode == 'frame':
        img_name = os.path.basename(path).replace('.npy', '.png')
        img_path = os.path.join(os.path.dirname(os.path.dirname(path)), img_name) if save is None else os.path.join(save, img_name)
        
    else:
        raise ValueError(f'Invalid mode: {mode}')

    # Setup the scene (lights / render engine / resolution etc)
    setup_scene(res=res, denoising=denoising, oldrender=oldrender, accelerator=accelerator, device=device)

    data = npy2obj(path, smpl)

    print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
    for frame_i in tqdm(range(data.nframes)):
        data.save_obj(os.path.join(results_dir, 'frame{:03d}.obj'.format(frame_i)), frame_i)

    # Number of frames possible to render
    nframes = data.nframes

    lims = data.get_boader()
    minx, miny, minz = lims[0]
    # Create a floor
    plot_floor(lims, big_plane=True)
    data.load_sound_source(minz)

    # initialize the camera
    camera = Camera(first_root=data.get_root(0), mode=mode)

    frameidxs = get_frameidx(mode=mode, nframes=nframes,
                            exact_frame=exact_frame,
                            frames_to_keep=num)
    if mode == "sequence":
        traj_frameidxs = get_frameidx(mode=mode, nframes=nframes,
                        exact_frame=exact_frame,
                        frames_to_keep=4*num)
        data.plot_traj(traj_frameidxs)

    nframes_to_render = len(frameidxs)

    imported_obj_names = []
    for index, frameidx in tqdm(enumerate(frameidxs)):
        mat = GT_SMPL if gt else GEN_SMPL
        if mode != "sep_sequence" and mode != "sequence":
            camera.update(data.get_root(frameidx))
        islast = index == (nframes_to_render - 1)
        path = os.path.join(results_dir, 'frame{:03d}.png'.format(frameidx))
        if mode == "sep_sequence":
            path = os.path.join(save_sep_dir, 'frame{:03d}.png'.format(frameidx))
        if mode == "sequence":
            alpha = 0.2 + 0.16 * index
            mat_tmp = mat.copy()
            mat_tmp.name = f"mat{frameidx}"
            mat_tmp.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = alpha
            mat_tmp.blend_method = 'BLEND'
            mat_tmp.shadow_method = 'HASHED'
            obj_names = data.load_in_blender(results_dir, frameidx, mat_tmp)
        else:
            obj_names = data.load_in_blender(results_dir, frameidx, mat)

        if mode != "sequence" or islast:
            render_path = path if mode != "sequence" else img_path
            render_current_frame(render_path)
            delete_objs(obj_names)

    # remove every object created
    delete_objs(imported_obj_names)
    delete_objs(["Plane", "myCurve", "Cylinder", "sound source"])

    if mode == "video":
        video = Video(results_dir, fps=fps)
        vid_path = results_dir.replace("_frames", ".mp4")
        vid_path = results_dir.replace("_frames", ".gt.mp4") if gt else results_dir.replace("_frames", ".pred.mp4")
        video.save(out_path=vid_path)

        audio_dir = "./sam/stereo_audio" if os.path.exists("./sam/stereo_audio") else "./sam/audio"
        audio_names = sorted(os.listdir(audio_dir))
        audio_name = parsed_name + '.wav' if os.path.exists("./sam/stereo_audio") else parsed_name + '_l.wav'
        if audio_name in audio_names:
            audio_dir_ = os.path.join(audio_dir, audio_name)
            name_w_audio = parsed_name + "_audio"
            vid_path_waudio = vid_path.replace(parsed_name, name_w_audio)
            cmd_audio = f"ffmpeg -i {vid_path} -i {audio_dir_} -map 0:v -map 1:a -c:v copy -shortest -y {vid_path_waudio} -loglevel quiet"
            os.system(cmd_audio)
            cmd_rm = f"rm {vid_path}"
            os.system(cmd_rm)
    
    elif mode == "sequence":
        print(f"Sequence generated at: {img_path}")

    else:
        print(f"Frame generated at: {img_path}")
    
    shutil.rmtree(results_dir)
    print(f"Tmp folder removed")

def render_cli() -> None:
    cfg = parse_args()

    if cfg.npy:
        paths = [cfg.npy]
    elif cfg.dir:
        paths = []
        file_list = os.listdir(cfg.dir)
        for item in file_list:
            if item.endswith(".npy"):
                paths.append(os.path.join(cfg.dir, item))
    else:
        raise ValueError(f'{cfg.npy} and {cfg.dir} are both None!')

    for path in paths:
        try:
            with open(path, 'rb') as f:
                npy = np.load(f, allow_pickle=True)

        except FileNotFoundError:
            print(f"{path} not found")
            continue

        render(
            path,
            mode=cfg.mode,
            smpl=cfg.smpl,
            gt=cfg.gt,
            exact_frame=cfg.exact_frame,
            num=cfg.num,
            always_on_floor=cfg.always_on_floor,
            denoising=cfg.denoising,
            oldrender=cfg.oldrender,
            res=cfg.res,
            accelerator=cfg.accelerator,
            device=cfg.device,
            fps=cfg.fps,
            save=cfg.save)

if __name__ == '__main__':
    render_cli()