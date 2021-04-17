# PSR-Net (WWW 2021 )
Progressive Semantic Reasoning for Image Inpainting

## Requirements
* tensorflow 2.0 （required）
* numpy 1.19.5 (optional)
* [loader](https://github.com/sfwyly/loader) 0.1 (required)
* opencv 4.1.1.26 (optional)
* Pillow 6.0.0 (optional)
* pathlib 1.0.1 (optional)

## Usage

> run model
```
  python train.py
```
> test model
```
  python test.py
```
All configuration option in config.py.

## Training and Fine Tuning

> training (Default)
```
  generator = build_model(mode = "training")
```
> fine tuning (Option)
```
  generator = build_model(mode = "tuning")
```

## Mask Dataset
We provide two ways to support loading the Mask dataset.
1. Existing Mask Dataset  
set *train_mask_path* and *val_mask_path* in config.py.
2. Generating Random Mask (we used the strategy of "gated conv" paper)  
set *generated_mask=False* in config.py

## Pretrained model
* Places2
* CelebA
* Paris StreetView

## Reference
```
Junjie Jin, Xinrong Hu, Kai He, Tao Peng, Junping Liu, and Jie Yang. 2021.Progressive Semantic Reasoning for Image Inpainting. InProceedings of theWeb Conference 2021 (WWW ’21 Companion), April 19–23, 2021, Ljubljana,Slovenia.ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3442442.3451142
```
