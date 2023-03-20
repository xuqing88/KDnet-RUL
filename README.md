# KDnet-RUL: A knowledge distillation framework to compress deep neural network for machine remaining useful lifte prediction
This code is for paper "KDnet-RUL: A knowledge distillation framework to compress deep neural networks for machine remaining learning useful life prediction

### Download the dataset

Please download the dataset from below link and put all files into `data` folder. 

https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

### Training Steps

1. Run `data_processing_train_valid_test.py` to generate the training / validation / test sets

2. Prepare the teacher models for knowledge distillation. Run `lstm_teacher.py` to generate LSTM teachers for each subset.

3. Train compact student with proposed KDnet_RUL, `cnn_student_gan+sequential_val.py`.
