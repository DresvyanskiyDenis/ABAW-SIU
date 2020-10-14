# An Audio-Video Deep and Transfer Learning Framework for Multimodal Emotion Recognition in the wild
In this paper, we present our contribution to the ABAW facial expression challenge. We report the proposed system and the official challenge results adhering to the challenge protocol. Using end-to-end deep learning and benefiting from transfer learning approaches, we reached the 3rd place in the Expression challenge with the test set performance measure of 42.10%.

The paper is available via this [link](https://arxiv.org/abs/2010.03692).

The weights for models and support files can be downloaded through this [link](https://drive.google.com/drive/folders/1Sw_Zgp0rCKEVVlH0bjUXESn3QMpBRds-?usp=sharing).


Every model from the article is located in separate folders with all needed code for generating, training, and predicting processes.

The code for generating and using models organized as follows:
+ Video-based models
  + VGGFace2-based model: Resnet50 pretrained on the [VGGFace2 dataset](https://arxiv.org/abs/1710.08092). 
+ Audio-based models
  + 1D CNN + LSTM model: Our own created 1D CNN + LSTM model, which is able to model sequence-to-one processing with capturing temporal information.
  + PANN-based model: [Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/abs/1912.10211)
