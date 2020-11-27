
# Important

Due to the limitation of the size of the uploaded files on GitHub,we saved the records of our model training in baidu's network disk .You can use this [link](https://pan.baidu.com/s/1C0TmImwmb1PjtdomnC9sdw) and verification code is **96qk**  directly to get the weights we used in the final round.

If you need to get the same score as out semi-final through the model parameter file given by us,please directly load the file into the corresponding configuration file and merge it.  

# NOTION

We used data sets other than those given in the semi-final.Here are the names and paths of the additional data sets we used.

- [x] [NAIC 2019 preliminary contestand NAIC 2019 intermediary heat](https://problemconfig-1256691515.cos.ap-guangzhou.myqcloud.com/10/REID2019.zip?q-sign-algorithm=sha1&q-ak=AKIDZs63jAnixyqxpcoHIdjXT8IrEQM0MUKu&q-sign-time=1606392720%3B1606393680&q-key-time=1606392720%3B1606393680&q-header-list=&q-url-param-list=&q-signature=39d632dfc5655c24f6d0db49808c9eeb32ab1d07)  
- [x] [NAIC 2020 preliminary contest](https://awscdn.datafountain.cn/cometition_data2/Files/PengCheng2020/ReID/train.zip)  
- [x] [NAIC 2020 intermediary heat](http://datafountain.int-yt.com/Files/PengCheng2020/ReID/fusai/train.zip)  

# Datafountain_Person_ReID_Competition

**This repository contains our source code  for the Person ReID Compitition of Datafountain. We are team,神圣兽国游尾郡窝窝乡独行族妖侠 , who ranked 97st in A and 34th in B.**

## Authors

- [Yin Liu](https://github.com/ly-wenli)
- [Bi Li]()
- [Feng Cheng](https://github.com/Chase-code)
- [JunJie Wang](https://github.com/guxinghaoyun)   


## Introduction

Detailed information about the Person ReID Compitition of NAIC can be found [here](https://www.datafountain.cn/competitions/454).

The code is modified from [fast_reid](https://github.com/JDAI-CV/fast-reid) and [NAIC_Person_ReID_DMT](https://github.com/heshuting555/NAIC_Person_ReID_DMT)

## Useful Tricks

- [x] RGB convert CMYK
- [x] DataAugmention(RandomErase + ColorJittering +RandomAffine + RandomHorizontallyFlip + Padding + RandomCrop)
- [x] WarmUp + MultiStepLR 
- [x] Ranger
- [x] ArcFace
- [x] Split Reranking
- [x] Gem
- [x] Weighted Triplet Loss
- [x] Remove Long Tail Data (pid with single image)
- [x] Distmat Ensemble
- [x] Freeze backbone layers

1. Due to the characteristics of the dataset, we find that the convergence accuracy will be improved after the data set is converted from RGB to CMYK color domain.Although we didn't have time to try it in this competition, it proved to be a very useful thing in this competitions.     
2. Due to the characteristics of the dataset, we find color Jittering can greatly improve model performance. 
3. Because the number of gallery in List B is too large,  Yin Liu have reconstructed the rerank data of list B, so that they can carry out normal training on a machine with less memory.
4. We use Ranger optimizer to make the model converge faster and better.
5. We found that freezing the entire backbone layer converges faster and better with fast_reid as the baseline.
6. FP16 training can save 30% memory cost and  is 30% faster with no precision drop. Although we didn't have time to try it in this competition, it proved to be a very useful thing in other competitions. Recommend everyone to use. You can refer to [apex](https://github.com/NVIDIA/apex) install it. if you don't have apex installed, please turn-off FP16 training by setting SOLVER.FP16=False 

### Project File Structure

```
+-- NAIC comp
|   +-- NAIC_ReID(put code here)
|   +-- model(dir to save the output)
|   +-- data
|		+-- MyDataSet
|			+--train_2019_cs
|				+--train
|				---train_list.txt
|			+--train_2019_fs
|				+--train
|				---train_list.txt
|			+--train_2020_cs
|				+--images
|				---train_list.txt
|			+--train_2020_fs
|				+--images
|				---train_list.txt
|			+--test
|				+--query
|				+--gallery
```



## Get Started

1. Because the backbone network we used in the semi-final is Resnet101_ibn,and we used its weight file obtained from the imagenet competiton in the training process,so you need to [download]() the weight file of this model.

2. `cd` to folder where you want to download this repo

3. Run `git clone https://github.com/ly-wenli/DF_Person_ReID.git`

4. Install dependencies:
   - [pytorch>=1.1.0](https://pytorch.org/)
   - python>=3.5
   - torchvision
   - [yacs](https://github.com/rbgirshick/yacs)
   - cv2
   
   We use cuda 10.1/python 3.8.3/torch 1.6.0/torchvision 0.7.0 for training and testing.
   
5.  [ResNet-ibn](https://github.com/XingangPan/IBN-Net) is applied as the backbone. Download ImageNet pretrained model  [here](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S) 

## RUN

1. If you want to get the same score as online in the Person ReID Compitition of Datafountain. Use the following commands:

   ```bash
   bash run.sh
   ```

2. If  you want to use our baseline for training. 

   ```bash
   python train.py --config_file [CHOOSE WHICH config TO RUN]
   # To get a B score of 34, you can use the following code.
   #python train.py --config_file configs/naic_round2_model_b.yml
   ```

4. If  you want to test the model and get the result in json format required by the competition.

   ```bash
   python test.py --config_file [CHOOSE WHICH CONFIG TO TEST]
   # For example, you can use the following code to directly get the same result as our B-list data.
   #python test.py --config_file configs/naic_round2_model_b.yml
   ```




