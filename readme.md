# PointLSTM

This repo holds the codes of paper:  [An Efficient PointLSTM for Point Clouds Based Gesture Recognition (CVPR 2020)](http://openaccess.thecvf.com/content_CVPR_2020/html/Min_An_Efficient_PointLSTM_for_Point_Clouds_Based_Gesture_Recognition_CVPR_2020_paper.html).

## Abstract

Point clouds contain rich spatial information, which provides complementary cues for gesture recognition. In this paper, we formulate gesture recognition as an irregular sequence recognition problem and aim to capture long-term spatial correlations across point cloud sequences. A novel and effective PointLSTM is proposed to propagate information from past to future while preserving the spatial structure. The proposed PointLSTM combines state information from neighboring points in the past with current features to update the current states by a weight-shared LSTM layer. This method can be integrated into many other sequence learning approaches. In the task of gesture recognition, the proposed PointLSTM achieves state-of-the-art results on
two challenging datasets (NVGesture and SHREC’17) and outperforms previous skeleton-based methods. To show its advantages in generalization, we evaluate our method on MSR Action3D dataset, and it produces competitive results with previous skeleton-based methods.

## Prerequisites

These code is implemented in Pytorch (>1.0). Thus please install Pytorch first.
## Usage

### Get the code

Clone this repo with git, please use:
```git
git clone https://github.com/Blueprintf/pointlstm_gesture_recognition_pytorch.git
```
### Data Preparation
- Download the [SHREC'17 dataset](http://www-rech.telecom-lille.fr/shrec2017-hand/) and put `HandGestureDataset_SHREC2017` directory to `./dataset/SHREC2017.` It is suggested to make a soft link toward downloaded dataset.
- Generate point cloud sequences from depth video, and save the processed point clouds in ```./dataset/Processed_SHREC2017```. Each video generate 32*256 points, and the generated point clouds occupy about 2.5G.
```python
cd dataset
python shrec17_process.py
```
### Training

Training of the PointLSTM-middle with k=16 on SHREC'17:

```python
cd experiments
python main.py --phase=train --work-dir=PATH_TO_SAVE_RESULTS --device=0 
```
We also provided trained model at here [Google Drive](https://drive.google.com/file/d/1eC4x9T1GXeS5SurxeFzBVkRSa1iFb9Gk/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1yryfRaN0NFW5eIIg5Uj67A ) [passwd: trhi].

### Inference

```python
cd experiments
python main.py --phase=test --work-dir=PATH_TO_SAVE_RESULTS --device=0 --weights=PATH_TO_WEIGHTS
```
### Citation

Please cite the following paper if you feel PointLSTM useful to your research.

```latex
@inproceedings{min_CVPR2020_PointLSTM,
  title={An Efficient PointLSTM for Point Clouds Based Gesture Recognition},
  author={Min, Yuecong and Zhang, Yanxiao and Chai, Xiujuan and Chen, Xilin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5761--5770},
  year={2020}
}
```

### Reference

[1] Yan, Sijie, Yuanjun Xiong, and Dahua Lin. "Spatial temporal graph convolutional networks for skeleton-based action recognition." Thirty-second AAAI conference on artificial intelligence. 2018. [[pdf]](http://www.dahualin.org/publications/dhl18_stgcn.pdf) [[code]](https://github.com/yysijie/st-gcn)