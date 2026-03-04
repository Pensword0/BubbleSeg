# BubbleSeg

BubbleSeg is a DeepLabV3+-based semantic segmentation framework for identifying bubble and solid phases in fluidized-bed images.

## Features

- DeepLab-based segmentation with configurable backbone (default: ResNet18)
- VOC-style dataset loading and augmentation pipeline
- Multi-scale and sliding-window inference modes
- Automatic export of segmentation masks and overlay visualizations

## Project Structure

```text
BubbleSeg/
|-- base/
|-- data/
|   `-- BubbleDataSet/
|       |-- JPEGImages/
|       |-- SegmentationClass/
|       `-- ImageSets/
|           `-- Segmentation/
|-- dataloaders/
|-- models/
|-- pth/
|   |-- best_model.pth
|   `-- config.json
|-- test/
|-- config.json
|-- train.py
`-- inference.py
```

## Dataset

The **BubbleDataSet** used in this project is publicly available on Zenodo:  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18849072.svg)](https://doi.org/10.5281/zenodo.18849072)

Place the dataset under `./data/BubbleDataSet` in VOC format (`JPEGImages`, `SegmentationClass`, and `ImageSets/Segmentation`).

If you use this dataset in research or production, please cite the DOI and follow the license and usage terms on the Zenodo page.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- scipy
- Pillow
- tqdm
- opencv-python

Example installation:

```bash
pip install torch torchvision numpy scipy pillow tqdm opencv-python
```

## Training

Train with the project config:

```bash
python train.py --config config.json
```

Resume from a checkpoint:

```bash
python train.py --config config.json --resume path/to/checkpoint.pth
```

Select specific GPU devices (optional):

```bash
python train.py --config config.json --device 0
```

## Inference

Run inference with the released checkpoint:

```bash
python inference.py --config ./pth/config.json --model ./pth/best_model.pth --test ./test --output outputs --extension jpg
```

Switch inference mode (optional):

```bash
python inference.py --config ./pth/config.json --model ./pth/best_model.pth --test ./test --mode sliding
```

## Output

Inference results are saved to:

- `outputs/masks`: colorized segmentation masks
- `outputs/overlays`: side-by-side original image and mask overlay

## Acknowledgement

This project is developed based on the open-source repository **pytorch-segmentation** by yassouali:  
https://github.com/yassouali/pytorch-segmentation

Parts of the source code are simplified and modified for bubble/solid phase segmentation in fluidized-bed systems.

## Contact

ybwj990122@163.com
