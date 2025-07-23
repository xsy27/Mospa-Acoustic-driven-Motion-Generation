import os
import numpy as np
import torch
from utils.nn_transforms import neural_FK

NJOINTS = 25

def smplx_forward(smplx, bs, nframes, nposes, ntransls, nbetas):
    pred_xyz = []
    nposes, ntransls, nbetas = nposes.float(), ntransls.float(), nbetas.float()
    for i in range(bs):
        smplx_pose = nposes[i]
        smplx_transl = ntransls[i]
        smplx_betas = nbetas[i].reshape(1, -1)
        # print(smplx_pose.shape, smplx_transl.shape)

        global_orient = smplx_pose[:, :3]
        body_pose = smplx_pose[:, 3:66]
        jaw_pose = smplx_pose[:, 66:69]
        leye_pose = smplx_pose[:, 69:72]
        reye_pose = smplx_pose[:, 72:75]
        # left_hand_pose = smplx_pose[:, 75:120]
        # right_hand_pose = smplx_pose[:, 120:]
        left_hand_pose = None
        right_hand_pose = None
        body_parms = {
            'global_orient': global_orient,'body_pose': body_pose,
            'jaw_pose': jaw_pose,'leye_pose': leye_pose,
            'reye_pose': reye_pose,'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,'transl': smplx_transl,
            'betas': smplx_betas}
        keypoints = smplx.forward(**body_parms).joints[:, 0:NJOINTS, :]

        global_shift = keypoints[:, 0:1, :]
        keypoints = keypoints - global_shift
        keypoints = keypoints + smplx_transl.reshape(nframes, 1, 3) # [nframes, njoints, 3]

        pred_xyz.append(keypoints)

    return pred_xyz

def get_smplx_joints(model_path='smpl_models/smplx', gender='NEUTRAL_2020', njoints=55):
    model_name = f"smplx_{gender}".upper()
    smplx_path = os.path.join(model_path, f"{model_name}.npz")
    if(not os.path.exists(smplx_path)): raise(ValueError("smplx model not found"))

    smplx_data = np.load(smplx_path, allow_pickle=True)
    parents = smplx_data['kintree_table']
    J_regressor = smplx_data['J_regressor']
    v_template = smplx_data['v_template']

    parents[0][0] = -1
    parents = parents[:, :njoints]

    J = J_regressor @ v_template
    J = J[:njoints]

    skeletons = np.zeros((njoints, 3))
    skeletons[0] = J[0]
    for i, pi in enumerate(parents[0]):
        if pi == -1:
            assert i == 0
            continue
        skeletons[i] = J[i] - J[pi]
    
    return torch.from_numpy(skeletons).float(), parents

def smplx_FK(skeletons, parents, bs, nposes, ntransls):

    pred_xyz = neural_FK(nposes, skeletons, ntransls, parents, rotation_type='aa')

    return pred_xyz

if __name__ == '__main__':
    skeletons, parents = get_smplx_joints(model_path="smpl_models/smplx", gender="NEUTRAL_2020", njoints=25)
    if skeletons is not None and parents is not None: print(skeletons.shape, parents.shape)