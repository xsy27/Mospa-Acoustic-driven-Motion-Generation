import os
import numpy as np

"""SAM Dataset Loader"""

class SAMDataset():
    
    def __init__(self, data_dir):
        assert os.path.exists(data_dir), f'Data does not exist at {data_dir}!'

        self.NUM_JOINTS = 55
        self.audio_dir = os.path.join(data_dir, 'audio/')
        self.motion_dir = os.path.join(data_dir, 'motion/')
        self.filter_file = os.path.join(data_dir, 'ignore_list.txt')

    def load_audio(self, seq_name):
        """Load an audio in wav format"""
        pass

    def load_motion(self, seq_name):
        """Load a motion sequence represented using SMPLX format."""
        file_path = os.path.join(self.motion_dir, f'{seq_name}.npz')
        assert os.path.exists(file_path), f'File {file_path} does not exist!'
        with open(file_path, 'rb') as f:
            data = np.load(f)
            smpl_betas = data['betas']  # (300, )
            smpl_poses = data['poses']  # (N, 55, 3)
            smpl_trans = data['trans']  # (N, 3)
            sound_sources = data['sound_source'] # (N, 3)
        return smpl_trans, smpl_poses.reshape(-1, self.NUM_JOINTS*3), smpl_betas, sound_sources