import os
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from textwrap import wrap
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)

NJOINTS = 25
height = 540
width = 960

def export(motions, rotations, sound_sources, names, save_path, betas=None, prefix=None):
    # motions: [nmotions, nframes, NJOINTS, 3]
    # rotations: [nmotions, nframes, NJOINTS, 3]

    assert len(names) == len(motions)
    if rotations is not None: assert len(rotations) == len(motions)
    assert len(sound_sources) == len(motions)

    write2npy(motions, rotations, sound_sources, names, betas, save_path)
    # visualize3d(save_path, prefix, names, motions)

def write2npy(motions, rotations, sound_sources, motion_names, betas, expdir):
    assert len(motions) == len(motion_names)

    ep_path = os.path.join(expdir, "npy")
        
    if not os.path.exists(ep_path):
        os.makedirs(ep_path)

    for i in tqdm(range(len(motions)),desc='Generating npy'):
        np_motion = motions[i]
        np_rotation = rotations[i] if rotations is not None else None
        np_sound_source = sound_sources[i]
        np_betas = betas[i] if betas is not None else np.zeros(300, dtype=np.float32)
        npy_data = {"position": np_motion, "rotation": np_rotation, "sound_source": np_sound_source, "betas": np_betas}

        motion_path = os.path.join(ep_path, motion_names[i])
        np.save(motion_path, npy_data)

def visualize3d(save, prefix, names, motions, figsize=(9.6, 6.4), fps=30, radius=5):

    def init():
        fig.suptitle(title, fontsize=8)
    
    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)
    
    def update(index):
        ax.clear()
        ax.set_xlim3d(-radius / 2, radius / 2)
        ax.set_ylim3d(-radius / 2, radius / 2)
        ax.set_zlim3d(-radius / 3., radius * 2 / 3.)
        ax.grid(b=False)
        ax.view_init(elev=120, azim=-90, roll=0)
        ax.dist = 7.5

        plot_xzPlane(MINS[0]-radius/4, MAXS[0]+radius/4, 0, MINS[2]-radius/4, MAXS[2]+radius/4)
        used_colors = colors

        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
    video3d_dir = os.path.join(save, 'videos3d')
    if not os.path.exists(video3d_dir):
        os.makedirs(video3d_dir)
    kinematic_tree = [
        [0, 1, 4, 7, 10],
        [0, 2, 5, 8, 11],
        [0, 3, 6, 9, 12, 15],
        [9, 13, 16, 18, 20],
        [9, 14, 17, 19, 21]
    ]
    matplotlib.use('Agg')

    for i, motion in enumerate(tqdm(motions, desc='Generating 3D animation')):
        name = names[i].split('.')[0]
        title = title = '\n'.join(wrap(name, 40))
        data = motion.copy() # [nframes, NJOINTS, 3]
        data *= 1
        assert (data.shape == motion.shape)

        fig = plt.figure(figsize=figsize)
        # plt.tight_layout()
        ax = fig.add_subplot(projection='3d')
        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)
        init()
        colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
        colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
        colors = colors_orange
        if(prefix == 'gt'):
            colors = colors_blue
        
        frame_num = data.shape[0]

        height_offset = MINS[1]
        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]

        data[..., 0] -= data[0:1, 0:1, 0]
        data[..., 2] -= data[0:1, 0:1, 2]

        ani = FuncAnimation(fig, update, frames=frame_num, interval=240, repeat=False)
        video3d_file_name = name + '.' + prefix + '.mp4'
        video3d_file_path = os.path.join(video3d_dir, video3d_file_name)
        ani.save(filename=video3d_file_path, writer="ffmpeg", fps=fps)

        audio_name = name + '_l.wav'
        audio_dir = "sam/audio"
        audio_names = sorted(os.listdir(audio_dir))
        
        if audio_name in audio_names:
            audio_dir_ = os.path.join(audio_dir, audio_name)
            name_w_audio = name + "_audio"
            cmd_audio = f"ffmpeg -i {video3d_file_path} -i {audio_dir_} -map 0:v -map 1:a -c:v copy -shortest -y {video3d_dir}/{name_w_audio}.{prefix}.mp4 -loglevel quiet"
            os.system(cmd_audio)
            cmd_rm = f"rm {video3d_dir}/{name}.{prefix}.mp4"
            os.system(cmd_rm)

        plt.close()