Kaggle TensorFlow Speech Recognition Challenge
==========================

This repository contains my solution to the Kaggle
[TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge).
The competition's goal was to train a model to recognize ten simple spoken words using Google Brain's speech command 
data set. 

I wrote my solution in Python and TensorFlow. As a learning exercise,
I tried designing my own network rather than starting with one of the
well known architectures. I started with a pure RNN approach but later
added 2D convolutional layers as they were effective in finding
structure within the [Mel Frequency Cepstral Coefficients](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
transformation of the audio signals. My final design was:

| Layer | Type | Description |
|-------|------|-------------|
| Input | Data | MFCC transformed data |
| Conv1 | Conv2d | 64 filters, 10x5 kernel, 1x2 strides |
| BatchNorm1 | Batch Norm | Momentum 0.9 |
| Relu1 | Relu | Relu activation layer |
| MaxPool1 | MaxPool | 3x3 kernel, 1x2 strides |
| DropOut1 | Dropout | Rate 0.5 |
| Conv2 | Conv2d | 128 filters, 10x5 kernel, 1x2 strides |
| BatchNorm2 | Batch Norm | Momentum 0.9 |
| Relu2 | Relu | Relu activation layer |
| MaxPool2 | MaxPool | 3x3 kernel, 1x2 strides |
| DropOut2 | Dropout | Rate 0.5 |
| Conv3 | Conv2d | 256 filters, 10x5 kernel, 1x1 strides |
| BatchNorm3 | Batch Norm | Momentum 0.9 |
| Relu3 | Relu | Relu activation layer |
| MaxPool3 | MaxPool | 3x5 kernel, 1x1 strides |
| DropOut3 | Dropout | Rate 0.5 |
| RNN1 | GRU | 128 units |
| FC1 | Dense | 256 units |
| BatchNorm 4 | Batch Norm | Momentum 0.9 |
| Relu4 | Relu | Relu activation layer |
| FC2 | Dense  | Final output logits |

I trained the model for 30 epochs at a batch size of 512 on my NVIDIA TitanX GPU. I used the
[Python Speech Features](https://github.com/jameslyons/python_speech_features) package to transform
the data from audio to MFCC. My final score was
80.8% accuracy which placed me 479th out of 1315 teams (top 37%). 
