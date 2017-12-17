# SRCNN-Keras

This code completes Super Resolution using Convolutional Neural Network. It is based on keras platform. For time saving, it decomposes images into blocks and handle them seperately. So we need to preprocess the images into several 32\*32 blocks.

1. First run 'create_data.ipynb' to create block dataset for training and testing.

2. Run 'train.ipynb'. This code is used for training the convolutional neural network model and will save the model into 'SRCNN_model.h5'

3. Run 'reconstruct.py' to test the model and reconstruct the image.
