
## Environment 
```
conda create -n hm-vae-env python=3.8
conda activate hm-vae-env
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install tqdm
pip install torchgeometry
pip install tensorboard
pip install scipy
pip install pyyaml
pip install opencv-python
pip install matplotlib
```

## Train hm-vae 
```
python train_motion_vae.py --config ./configs/len64_no_aug_hm_vae.yaml
```

## Motion Completeion
Coming soon. 

## Motion Interpolation 
Coming soon. 