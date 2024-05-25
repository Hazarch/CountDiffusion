# CountDiffusion

## Installation
cuda: 11.7
```bash
git clone https://github.com/Hazarch/CountDiffusion
cd CountDiffusion
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything

export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.7/
conda env create -f environment.yml

cd Grounded-Segment-Anything
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
```

## Download ckpts
Pixart-sigma: https://huggingface.co/PixArt-alpha/PixArt-Sigma/blob/main/PixArt-Sigma-XL-2-1024-MS.pth

Grounding DINO: https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

SAM: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

## Inference
Complete configs/sam_config.yaml and scripts/demo.py relative settings.
```bash
cd CountDiffusion
python scripts/demo.py --save_mid --MCS
```
