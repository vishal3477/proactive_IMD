# Proactive-image-manipulation-detection
Official Pytorch implementation of CVPR 2022 paper "Proactive Image Manipulation Detection ".

The paper and supplementary can be found at: 


![alt text](https://github.com/vishal3477/Reverse_Engineering_GMs/blob/main/image/teaser_resized.png?raw=true)
## Prerequisites

- PyTorch 1.5.0
- Numpy 1.14.2
- Scikit-learn 0.22.2

## Getting Started

## Datasets 
- Download the dataset for the corressponding GMs from https://drive.google.com/drive/folders/1ZKQ3t7_Hip9DO6uwljZL4rYAn5viSRhu?usp=sharing
- For leave out experiment, put the training data in train folder and leave out models data in test folder
- For testing on custom images, put the data in test folder.

## Pre-trained model
The pre-trained model trained on STGAN can be downloaded from: 

## Training
- Provide the train and test path in respective codes as sepecified below. 
- Provide the model path to resume training
- Run the code as showb below:

```
python pro_det_train.py
```



## Testing using pre-trained models
- Download the pre-trained models from https://drive.google.com/drive/folders/1bzh9Pvr7L-NyQ2Mk-TBSlSq4TkMn2anB?usp=sharing
- Provide the model path in the code
- Run the code as shown below:

```
python pro_det_test.py
```


If you would like to use our work, please cite:
```
@inproceedings{asnani2022proactive
      title={Proactive Image Manipulation Detection}, 
      author={Asnani, Vishal and Yin, Xi and Hassner, Tal and Liu, Sijia and Liu, Xiaoming},
      booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2022}
      
}
```
