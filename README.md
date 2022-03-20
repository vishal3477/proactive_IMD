# Proactive-image-manipulation-detection
Official Pytorch implementation of CVPR 2022 paper "Proactive Image Manipulation Detection ".

The paper and supplementary can be found at: 



## Prerequisites

- PyTorch 1.5.0
- Numpy 1.14.2
- Scikit-learn 0.22.2

## Getting Started

## Datasets 
- Every GM is used with different datasets they are trained on. Please refer to Table 2 of the supplementary for GM-dataset information. Download the dataset for the corressponding GMs from https://drive.google.com/file/d/1fAS7Sj3FhS6v31Z2hb9mp9gaGavgnLu5/view?usp=sharing
- The training data is used as CELEBA-HQ which is provided in the above link as CELEBA_HQ_TRAIN folder.

## Pre-trained model
The pre-trained model trained on STGAN can be downloaded from: https://drive.google.com/file/d/1p9zETa9rCU0wx8wD5Ige2TbCL8WciV7o/view?usp=sharing

## Training
- Provide the train and test path in respective codes as sepecified below. 
- Provide the model path to resume training
- Run the code as showb below:

```
python pro_det_train.py
```



## Testing using pre-trained models
- Download the pre-trained models
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
