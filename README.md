# Bioinfor DanQ
DanQ is a hybrid convolutional and recurrent deep neural network for quantifying the function of DNA sequences.
This is implemented by tensorflow-2.0 again.

## DanQ

### Model Architecture
CNN + BidLSTM + Dense

### Loss Function
Binary Cross Entropy

### Optimization Method
Adam

## USAGE

### Requirement
We run training on Ubuntu 18.04 LTS with a GTX 1080ti GPU.

[Python](<https://www.python.org>) (3.7.3) | [Tensorflow](<https://tensorflow.google.cn/install>) (2.0.0)
| [CUDA](<https://developer.nvidia.com/cuda-toolkit-archive>) (10.0) | [cuDNN](<https://developer.nvidia.com/cudnn>) (7.6.0)


### Data
You need to first download the training, validation, and testing sets from DeepSEA. You can download the datasets from 
[here](<http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz>). After you have extracted the
contents of the tar.gz file, move the 3 .mat files into the data/ folder. 

### Model
The model that trained by myself is available in BAIDU Net Disk [here](https://pan.baidu.com/s/1LiUAaEy5IlFDecl4DWXrKw)

### Preprocess
Because of my RAM limited, I firstly transform the train.mat file to .tfrecord files.
```
python preprocess.py
```

### Training
Then you can train the model initially.
```
CUDA_VISIBLE_DEVICES=0 python main_DanQ.py -e train
```

### Test
When you have trained successfully, you can evaluate the model.
```
CUDA_VISIBLE_DEVICES=0 python main_DanQ.py -e test
```

## RESULT
Yon can get the result in the **`./result/`** directory.

### Loss Curve
For DanQ:

![DanQ loss](./result/DanQ/model_loss.jpg 'DanQ Loss Curve')

For DanQ-JASPAR:

![DanQ-JASPAR loss](./result/DanQ_JASPAR/model_loss.jpg 'DanQ-JASPAR Loss Curve')

### Metric
We use two metrics to evaluate the model. (AUROC, AUPR)

For DanQ:

-|DNase|TFBinding|HistoneMark|All
:-:|:-:|:-:|:-:|:-:
AUROC|0.9022|0.9317|0.8303|0.9162
AUPR|0.4072|0.2984|0.3373|0.3176

For DanQ-JASPAR:

-|DNase|TFBinding|HistoneMark|All
:-:|:-:|:-:|:-:|:-:
AUROC|0.9124|0.9451|0.8395|0.9287
AUPR|0.4323|0.3271|0.3508|0.3441

## REFERENCE
> [DanQ: a hybrid convolutional and recurrent deep neural network for quantifying the function of DNA sequences](<https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4914104/>) | [Github](<https://github.com/uci-cbcl/DanQ/>)