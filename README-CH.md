# Cat and Dog CNN with ResNet50

### 資料集來源：[Kaggle - Cat and Dog](https://www.kaggle.com/datasets/tongpython/cat-and-dog)

**注意**：由於資料集過大，無法直接包含在此。請從提供的 Kaggle 連結自行下載。

## Overview

- Language: Python v3.10.12
- Package: Tensorflow
- Model: CNN(ResNet50)
- Loss Function: Cross Entropy
- Optimizer: Adam, Learning Rate = 0.0001
- data augmentation to reduce overfitting

## Model Architecture

```
              OPERATION        DATA DIMENSIONS   WEIGHTS(N)   WEIGHTS(%)

              Input   #####       3  224  224
         InputLayer     |      ----------------          0         0.0%
                      #####       3  224  224
      ResNet50 (Base)  \|/     ----------------    2359808         1.7%
               -      #####     512  224  224
       MaxPooling2D   Y max    ----------------          0         0.0%
                      #####     512  112  112
      Convolution2D    \|/     ----------------     147584         0.1%
               relu   #####     128  112  112
       MaxPooling2D   Y max    ----------------          0         0.0%
                      #####     128   56   56
           Flatten    |||||    ----------------          0         0.0%
                      #####         50176
              Dense   XXXXX    ----------------    1605696        74.3%
               relu   #####          32
           Dropout    |||||    ----------------          0         0.0%
                      #####          32
              Dense   XXXXX    ----------------         64         2.8%
               relu   #####           2
              Dense   XXXXX    ----------------         64         2.8%
            softmax   #####           2
```

## Conclusion

### Loss

![Loss](./image/loss.png)

### Accuracy

![Accuracy](./image/accuracy.png)



### Confusion Matrix - Accuracy Rate 97.53%

![image](./image/confusion_matrix.png)