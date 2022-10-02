# Decription
* The code is the official implementation of paper [Self-Ensemble Protection: Training Checkpoints Are Good Data Protectors](https://openreview.net/forum?id=9MO7bjoAfIA)
* This repository supports data protection on CIFAR-10, CIFAR-100, ImageNet subset
* protecting DNN and appropriator DNN: ResNet18, SENet18, VGG16, DenseNet121, GoogLeNet
* The experiments are run in an NVIDIA A100 GPU, but could modify the batch size to run on small GPUs
* Install dependencies
```
conda env create -f pt.yaml
```


# Reproduction
* train the protecting DNN for CIFAR-10, CIFAR-100, ImageNet subset

```
python vanilla.py
```
```
python vanilla100.py
```
```
python vanillaimg.py
```

* crafting protective samples (CIFAR-10, SEP)

```
python ens.py --num_model=30 --eps=2 --target_batch=0
```

* crafting protective samples (CIFAR-10, SEP-FA)

```
python ens_feature.py --num_model=30 --eps=2 --target_batch=0
```

* crafting protective samples (CIFAR-10, SEP-FA-VR)

```
python ens_feature_svre.py --num_model=15 --eps=2 --target_batch=0
```

* crafting protective samples (CIFAR-100, SEP-FA-VR)

```
python ens_feature_svre100.py --num_model=15 --eps=2 --target_batch=0
```

* crafting protective samples (ImageNet subset, SEP-FA-VR)

```
python ens_feature_svreimg.py --num_model=15 --eps=2 --target_batch=0
```

* train the appropriator DNN
```
python vanilla.py --uledir=samples/XX --eps=2
```
```
python vanilla100.py --uledir=samples/XX --eps=2
```
```
python vanillaimg.py --uledir=samples/XX --eps=2
```

# Files
```
├── ens_feature.py
├── ens_feature_svre100.py
├── ens_feature_svreimg.py
├── ens_feature_svre.py
├── ens.py
├── models
│   ├── densenet.py
│   ├── dpn.py
│   ├── efficientnet.py
│   ├── googlenet.py
│   ├── __init__.py
│   ├── lenet.py
│   ├── mobilenet.py
│   ├── mobilenetv2.py
│   ├── pnasnet.py
│   ├── preact_resnet.py
│   ├── regnet.py
│   ├── resnet.py
│   ├── resnext.py
│   ├── senet.py
│   ├── shufflenet.py
│   ├── shufflenetv2.py
│   └── vgg.py
├── pt.yaml
├── README.md
├── utils
│   ├── data.py
│   ├── __init__.py
│   ├── output.py
│   └── tmp.py
├── vanilla100.py
├── vanillaimg.py
└── vanilla.py
```
