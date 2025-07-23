# MOSPA

## Abstract

Enabling virtual humans to dynamically and realistically respond to diverse auditory stimuli remains a key challenge in character animation, demanding the integration of perceptual modeling and motion synthesis. Despite its significance, this task remains largely unexplored. Most previous works have primarily focused on mapping modalities like speech, audio, and music to generate human motion. As of yet, these models typically overlook the impact of spatial features encoded in spatial audio signals on human motion. To bridge this gap and enable high-quality modeling of human movements in response to spatial audio, we introduce the first comprehensive Spatial Audio-Driven Human Motion (SAM) dataset, which contains diverse and high-quality spatial audio and motion data. For benchmarking, we develop a simple yet effective diffusion-based generative framework for human MOtion generation driven by SPatial Audio, termed MOSPA, which faithfully captures the relationship between body motion and spatial audio through an effective fusion mechanism. Once trained, MOSPA could generate diverse realistic human motions conditioned on varying spatial audio inputs. We perform a thorough investigation of the proposed dataset and conduct extensive experiments for benchmarking, where our method achieves state-of-the-art performance on this task. Our model and dataset will be open-sourced upon acceptance. Please refer to our supplementary video for more details.

## Prerequisites

### 1. Environment setup
Create and activate the `mospa` conda environment using the `environment.yml` file
```shell
git clone https://github.com/xsy27/MOSPA.git
cd MOSPA
conda env create -f environment.yml
conda activate mospa
```

## Training

### 1. Prepare data
```shell
python -m data.prepare
```

### 2. Train the MOSPA network
```shell
python -W ignore -m main.train -n mospa -c ./config/mospa.json --epoch 6000 --batch_size 128
```
You may need to adjust the batch size to fit the RAM of your GPU.

### 3. Evaluate the MOSPA network
```shell
python -W ignore -m main.evaluate --ckpt ./save/mospa/weights_6000.pt --save_pred True -c ./config/mospa.json --model mospa
```
The visualized outputs are store in `./eval/mospa`

### 4. Check the training process
The checkpoints and the training process are store in save/ssl and save/mospa_afm_train.
Run
```shell
tensorboard --logdir=./save/mospa
```
to view the training trend.

## Evaluation

Simple metrics are stored in `./eval/mospa/metrics.txt`, including MPJPE, M-MPJPE, PA_MPJPE, R-MPJPE, and APD.

To calculate FID, R-precision, and Diversity, train the feature extractor first
```shell
python -m main.train_extractor
```

Then Run
```shell
python -m data.extract --src_dir ./eval/mospa/pred/npy --model mospa
python -m utils.metrics_v2
```

The results can be viewed at `metrics_v2.out`

## Visualization

## 1. Download Blender
The version of blender used in this project is 2.93.
You can download it from https://download.blender.org/release/Blender2.93/
Move it to the MOSPA directory.

## 2. Install python packages for Blender
```shell
./blender/2.93/python/bin/python3.9 -m ensurepip --upgrade

./blender/2.93/python/bin/python3.9 -m pip install torch==2.4.0 smplx trimesh tqdm moviepy==1.0.3 scipy chumpy --target=./blender/2.93/python/lib/python3.9/site-packages

./blender/2.93/python/bin/python3.9 -m pip install -U numpy==1.23.1
```

## 3. Render the motions
To render an animation:
```shell
./blender/blender --background --python main/render.py -- --npy `[path to npy file]` --mode video --gt --fps 30 > render_process.out
```
To render a sequence image:
```shell
./blender/blender --background --python main/render.py -- --npy `[path to npy file]` --mode sequence --gt --fps 30 > render_process.out
```
Remove the `--gt` option if you want to render the predicted motions/motion sequences.

## Acknowledgement

This project is significantly inspired by the following foundational works. We extend our sincere gratitude to them and encourage readers to cite these projects if they find our work useful:

- [CAMDM](https://github.com/AIGAnimation/CAMDM)
- [Bailando](https://github.com/lisiyao21/Bailando)
- [MotionLCM](https://github.com/Dai-Wenxun/MotionLCM)
- [MDM](https://github.com/GuyTevet/motion-diffusion-model)
- [Guided-Diffusion](https://github.com/openai/guided-diffusion)

## Copyright Information

The source code is released under the MIT License.