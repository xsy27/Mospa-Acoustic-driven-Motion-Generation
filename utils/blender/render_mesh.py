import argparse
import os
import numpy as np
import torch
import shutil
from tqdm import tqdm
from smplx import SMPL, SMPLX
from trimesh import Trimesh
import bpy
from utils.smplx import smplx_forward
from utils.nn_transforms import aa2mat
import sys
from scipy.spatial.transform import Rotation as R
np.set_printoptions(threshold=sys.maxsize)
NJOINTS = 25

class npy2obj:
    def __init__(self, npy_path, smpl_path):
        self.world_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

        self.npy_path = npy_path
        self.motions = np.load(self.npy_path, allow_pickle=True).item()['rotation'] # [frame_num, NJOINTS, 3]
        self.positions = np.load(self.npy_path, allow_pickle=True).item()['position'] # [frame_num, NJOINTS, 3]
        self.sound_sources = np.load(self.npy_path, allow_pickle=True).item()['sound_source'] # [frame_num, 3]
        self.betas = np.load(self.npy_path, allow_pickle=True).item()['betas'] # [300]
        root_orients = aa2mat(torch.from_numpy(self.motions[:, 0, :])).numpy()
        root_transls = self.positions[:, 0, :]
        root_transls = root_transls - np.array([root_transls[0, 0], 0, root_transls[0, 2]])

        self.sound_sources = np.array([
            self.world_mat @ (root_orient @ ss + root_transl) 
            for root_orient, ss, root_transl in zip(root_orients, self.sound_sources, root_transls)
        ])
        self.nframes, self.njoints, self.nfeats = self.motions.shape
        self.smplx = SMPLX(model_path=smpl_path,
                    use_face_contour=False,
                    gender='NEUTRAL_2020', 
                    num_betas=300,
                    num_expression_coeffs=100,
                    use_pca=False,
                    batch_size=self.nframes).eval()
        self.faces = self.smplx.faces

        self.rotations = torch.from_numpy(self.motions)
        self.global_orient = self.rotations[:, 0:1]
        self.rotations = self.rotations[:, 1:]
        # betas = torch.zeros([self.rotations.shape[0], self.smplx.num_betas], dtype=self.rotations.dtype, device=self.rotations.device)
        
        smpl_pose = torch.from_numpy(self.motions).reshape(self.nframes, -1)
        smpl_trans = torch.from_numpy(self.positions)[:, 0, :]
        smpl_trans = smpl_trans - np.array([smpl_trans[0, 0], 0, smpl_trans[0, 2]])
        smpl_betas = torch.from_numpy(self.betas).unsqueeze(0).float()
        global_orient = smpl_pose[:, :3]
        body_pose = smpl_pose[:, 3:66]
        jaw_pose = smpl_pose[:, 66:69]
        leye_pose = smpl_pose[:, 69:72]
        reye_pose = smpl_pose[:, 72:75]
        # left_hand_pose = smpl_pose[:, 75:120]
        # right_hand_pose = smpl_pose[:, 120:]
        left_hand_pose = None
        right_hand_pose = None
        body_parms = {
            'global_orient': global_orient,'body_pose': body_pose,
            'jaw_pose': jaw_pose,'leye_pose': leye_pose,
            'reye_pose': reye_pose,'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,'transl': smpl_trans,
            'betas': smpl_betas}
        self.out = self.smplx.forward(**body_parms)

        self.joints = self.out.joints.detach().cpu().numpy()[:, 0:NJOINTS, :]
        self.vertices = self.out.vertices.detach().cpu().numpy()
        self.minzs = self.get_minzs()
        # self.vertices += np.tile(self.positions[:, 0:1, :], (1, self.vertices.shape[1], 1))

        self.ground = True
    
    def get_trimesh(self, frame_i):
        # use this trick to avoid the ambguity cause by the smpl regression
        if self.ground:
            self.vertices = self.vertices - self.get_minzs()
        return Trimesh(vertices=self.vertices[frame_i], faces=self.faces)

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(frame_i)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path
    
    def get_root(self, i):
        return self.world_mat @ ((self.joints[i, 0].copy() - self.minzs[i, 0]) if self.ground else self.joints[i, 0].copy())
    
    def add_circle(self, r, color, transparency, location, name=""):
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=r,
            location=location,
            segments=32,
            ring_count=16,
        )
        sound_source = bpy.context.object
        material = bpy.data.materials.new(name="ssMaterial")
        material.use_nodes = True

        material.node_tree.nodes["Principled BSDF"].inputs['Alpha'].default_value = transparency
        material.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = color
        material.blend_method = 'BLEND'
        
        sound_source.data.materials.append(material)
        if name != "": sound_source.name = name
        bpy.ops.object.select_all(action='DESELECT')

    def load_sound_source(self, minz):
        location = self.sound_sources[0]
        color = (0.1, 0.1, 0.8, 1)
        self.add_circle(0.05, color, 1, location, "sound source")
        self.add_circle(0.2, color, 0.5, location)
        self.add_circle(0.5, color, 0.3, location)
        self.add_circle(2, color, 0.2, location)
    
    def plot_traj(self, idxs):
        color = (1, 0, 0, 1)
        for idx in idxs:
            self.add_circle(0.03, color, 1, self.get_root(idx))

    def load_in_blender(self, results_dir, i, mat):
        obj_path = os.path.join(results_dir, 'frame{:03d}.obj'.format(i))
        bpy.ops.import_scene.obj(filepath=obj_path)
        obj = bpy.context.selected_objects[0]
        print(f"Object {obj.name} imported")
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        # obj.location = self.get_root(i)
        obj.active_material = mat
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()
        bpy.ops.object.select_all(action='DESELECT')
        return obj.name

    def get_boader(self):
        shifted_vertices = (self.world_mat @ self.vertices.reshape(-1, 3).T).T.reshape(self.nframes, -1, 3)
        mins, maxs = np.min(shifted_vertices, axis=(0, 1)), np.max(shifted_vertices, axis=(0, 1))
        return (mins, maxs)

    def get_minzs(self):
        minzs = np.min(self.vertices, axis=1, keepdims=True)
        minzs[:, :, [0, 2]] = 0
        return minzs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", type=str, required=True)
    config = parser.parse_args()

    # config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # config.cuda = True if torch.cuda.is_available() else False
    config.smpl = 'smpl_models/smplx'

    assert config.npy_path.endswith('.npy')
    assert os.path.exists(config.npy_path)
    parsed_name = os.path.basename(config.npy_path).replace('.npy', '')
    results_dir_name = parsed_name + '_obj'
    results_dir = os.path.join(os.path.dirname(os.path.dirname(config.npy_path)), results_dir_name)
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    npy2obj = npy2obj(config.npy_path, smpl_path=config.smpl)

    print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
    for frame_i in tqdm(range(npy2obj.nframes)):
        npy2obj.save_obj(os.path.join(results_dir, 'frame{:03d}.obj'.format(frame_i)), frame_i)