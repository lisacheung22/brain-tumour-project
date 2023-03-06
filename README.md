# Project Core
In this project we worked to define a machine learning algorithm called convolutional neural network (CNN) to classify whether a MRI brain scan contains late-stage tumours. Our project core is to train a CNN model that classifies 2D images into either with tumours or without tumours to determine how accuracy can CNN be in recognizing abnormal MRI scans. For the project core, we used Brats MRI dataset containing 3000 images, in which we used 2400 for training our CNN model, 300 for validation during training, and 300 for testing performance. Dataset images are processed to be all grayscale with 256x256 pixels before feeding into Tensorflow CNN model during training.

Our model architecture deploys 11 layers in total, including 2 convolution layers each with 32 kernels to extract image features, 2 maxpooling layers and 2 dropout layers (rate = 0.25 and 0.5) to prevent overfitting. Model architecture is shown below: 

Model: "sequential"
_________________________________________________________________
Layer                     

=================================================================
 conv2d (Conv2D)             (None, 256, 256, 32)      320       
                                                                 
 batch_normalization (BatchN  (None, 256, 256, 32)     128       
 ormalization)                                                   
                                                                 
 max_pooling2d (MaxPooling2D  (None, 128, 128, 32)     0         
 )                                                               
                                                                 
 dropout (Dropout)           (None, 128, 128, 32)      0         
                                                                 
 conv2d_1 (Conv2D)           (None, 128, 128, 32)      9248      
                                                                 
 batch_normalization_1 (Batc  (None, 128, 128, 32)     128       
 hNormalization)                                                 
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 64, 64, 32)       0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 131072)            0         
                                                                 
 dense (Dense)               (None, 512)               67109376  
                                                                 
 dropout_1 (Dropout)         (None, 512)               0         
                                                                 
 dense_1 (Dense)             (None, 1)                 513       
                                                                 
=================================================================
Total params: 67,119,713
Trainable params: 67,119,585
Non-trainable params: 128
_________________________________________________________________

Our model was compiled with SGD optimizer and monitored with binary cross entropy loss function during training. The training process went for 14 epochs over 2400 images and took around 10 minutes. The batch size used in training was 32.

In the final epoch, it achieves 97.66% accuracy on validation data:
Epoch 14/14
75/75 [==============================] - 51s 673ms/step - loss: 0.0117 - accuracy: 0.9975 - val_loss: 0.0800 - val_accuracy: 0.9767

Then during model evaluation on 300 testing images, it achieves 99% accuracy on testing images:
10/10 [==============================] - 1s 130ms/step - loss: 0.0470 - accuracy: 0.9900


