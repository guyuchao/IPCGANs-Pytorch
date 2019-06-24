# IPCGANs-pytorch

> Author: Yuchao Gu

> E-mail: gyc.nankai@gmail.com

> Date: 2019-06-24

> Description: The code is an pytorch implementation of [《Face Aging with Identity-Preserved Conditional Generative Adversarial Networks》](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Face_Aging_With_CVPR_2018_paper.pdf)


---

## Usage

### Data Preparation

1. Download the [cacd-2000](https://bcsiriuschen.github.io/CARC/) datasets and unzip.
2. Use preprocess/preprocess_cacd.py to crop and align face region.
3. Use preprocess/gentraingroup.py to generate training list, please refer to data/cacd2000-lists as an example.

### Training

1. Pretrain Age Classify model

``` python
python pretrain_alexnet.py
```

2. Train IPCGANs

``` python
python IPCGANS_train.py
```

3. Pretrained IPCGANs and AgeAlexnet model can be downloaded [here](https://pan.baidu.com/s/1a6YkVMmT1HcGvgX0YLO9kQ), password:Lr6a

### Visualize

![](./readmeDisplay/1.png)

### Server

This project acted as an server for Android [demo](https://github.com/guyuchao/IPCGANs-Android). You can start flask server.
``` python
python app.py 
```

### Dependencies

This code depends on the following libraries:

* Python 3.6
* Pytorch 0.4.1
* PIL
* TensorboardX
* Flask

### Structure
```
IPCGANs
│
├── checkpoint  # path to store logging and parameters
│ 
├── data # path to specify training data groups and testing data groups
│ 
├── data_generator # pytorch dataloader for training
│ 
├── model # path to define model
│ 
├── utils
│
├── preprocess
│  
├── app.py # server scripts
│
├── demo.py # api for server 
│  
├── pretrain_alexnet.py
│
├── IPCGANS_train.py 
│
└── README.md # introduce to this project
```



