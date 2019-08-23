# Bioinfor DanQ
This is implemented by tensorflow-2.0 again.

## DanQ

### Model Architecture


### Loss Function
Binary Cross Entropy.

### Optimization Method
RMSProp.



## USAGE

### Requirement
We run training on Ubuntu 18.04 LTS with a GTX 1080ti GPU.

[Python](<https://www.python.org>) (3.7.3) | [Tensorflow](<https://tensorflow.google.cn/install>) (2.0.0beta1)
| [CUDA](<https://developer.nvidia.com/cuda-toolkit-archive>) (10.0) | [cuDNN](<https://developer.nvidia.com/cudnn>) (7.4.1)


### Data
You need to first download the training, validation, and testing sets from DeepSEA. You can download the datasets from 
[here] (<http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz>). After you have extracted the 
contents of the tar.gz file, move the 3 .mat files into the data/ folder. 

### Model
The model that trained by myself is available in BAIDU Net Disk [here] (https://pan.baidu.com/s/1tfYvDoO6Xvt7v7y70nDsXg)

### Preprocess
Because of my RAM limited, I firstly transform the train.mat file to .tfrecord files.
```
python preprocess.py
```

### Training
Then you can train the model initially.
```
CUDA_VISIBLE_DEVICES=0 python main_DanQ.py/main_DanQ_JASPAR.py -e train
```

### Test
When you have trained successfully, you can evaluate the model.
```
CUDA_VISIBLE_DEVICES=0 python main_DanQ.py/main_DanQ_JASPAR.py -e test
```

## RESULT
Yon can get the result in the **`./result/`** directory.

### Loss Curve

### Metric


## REFERENCE
> [DanQ: a hybrid convolutional and recurrent deep neural network for quantifying the function of DNA sequences](<https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4914104/>) | [Github](<https://github.com/uci-cbcl/DanQ/>)