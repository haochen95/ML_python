

# Goal  

Train a Neural Network to recognize digital numbers  

# Dataset:  

MINIST dataset contains image from 0-9 written by people with 28*28 pixels  
1. Training set: 60000 images = 50000 for training + 10000 for validation  
2. Test set: 10000 images

# Neural Network Design  

Input layer: flatten(28*28) = 784  
Hidden layer: 30  
Output layer: 10(0-9)  

# Apply SGD on NN  

```text  
Epoch 0  : 974 / 10000
Epoch 1  : 974 / 10000
Epoch 2  : 1135 / 10000
Epoch 3  : 1135 / 10000
Epoch 4  : 1135 / 10000
Epoch 5  : 1135 / 10000
Epoch 6  : 1135 / 10000
Epoch 7  : 1010 / 10000
Epoch 8  : 974 / 10000
Epoch 9  : 958 / 10000
Epoch 10  : 1009 / 10000
Epoch 11  : 1009 / 10000
Epoch 12  : 958 / 10000
Epoch 13  : 974 / 10000
Epoch 14  : 892 / 10000
Epoch 15  : 1028 / 10000
Epoch 16  : 982 / 10000
Epoch 17  : 974 / 10000
Epoch 18  : 982 / 10000
Epoch 19  : 1028 / 10000
Epoch 20  : 1010 / 10000
Epoch 21  : 982 / 10000
Epoch 22  : 1032 / 10000
Epoch 23  : 982 / 10000
Epoch 24  : 958 / 10000
Epoch 25  : 1010 / 10000
Epoch 26  : 1028 / 10000
Epoch 27  : 892 / 10000
Epoch 28  : 1032 / 10000
Epoch 29  : 892 / 10000
```

# Refernce  

* <http://neuralnetworksanddeeplearning.com/index.html>
* <https://medium.com/analytics-vidhya/neural-networks-for-digits-recognition-e11d9dff00d5>