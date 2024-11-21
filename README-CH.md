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

- **ResNet50（Feature Extractor）**：  
  - 預訓練於 ImageNet。  
  - 透過去除頂層全連接層，作為特徵提取器使用。

- **Convolutional Layer 1**:  
  - 512 個濾波器，3x3 核心。  
  - ReLU 激活函數。

- **Max Pooling Layer 1**:  
  - 2x2 池化大小，用於減少特徵圖大小。

- **Convolutional Layer 2**:  
  - 128 個濾波器，3x3 核心。  
  - ReLU 激活函數。

- **Max Pooling Layer 2**:  
  - 2x2 池化大小，用於減少特徵圖大小。

- **Flatten Layer**:  
  - 將 2D 特徵圖轉換為 1D 向量。

- **Dense Layer 1**:  
  - 32 個神經元。  
  - ReLU 激活函數。

- **Dropout Layer**:  
  - 丟棄 40% 的神經元，以防止過擬合。

- **Dense Layer 2 (Output Layer)**:  
  - 2 個神經元（分別對應貓和狗）。  
  - Softmax 激活函數。

## Conclusion

### Loss

![Loss](./image/loss.png)

### Accuracy

![Accuracy](./image/accuracy.png)



### Confusion Matrix - Accuracy Rate 97.53%

![image](./image/confusion_matrix.png)