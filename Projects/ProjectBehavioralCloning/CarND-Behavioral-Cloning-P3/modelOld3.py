import os
import csv
import cv2
import numpy as np
from scipy import ndimage
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage import io, color, exposure, filters, img_as_ubyte
from skimage.transform import resize
from skimage.util import random_noise


my_data = ['./dataayush']
udacity_data = ['./dataudacity'] 
               
# Code to retrieve driving data from driving_log.csv file
samples = []
# with open('/home/workspace/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
for list in [my_data, udacity_data]:
    for filename in list:
        with open(filename + '/driving_log.csv') as csvfile:
            # Line per line reading the data and appending to samples list
            for line in csv.reader(csvfile):
                samples.append(line)
                   
# If there is header in csv data, we donot want it to be uploaded into samples list, so we will use samples = samples[1:],
samples=samples[1:]   
print('Filenames loaded...')
# print(samples[1])
print(len(samples))
# image_datapath = "/home/workspace/CarND-Behavioral-Cloning-P3/data/IMG/"
image_datapath = "./data/IMG/"
# Each image dimension
row, col, ch = 160, 320, 3
Img_shape= (row, col, ch)
# Splitti ng the data saved in samples list into training and testing data. It is 80-20 split
train_samples, validation_samples = train_test_split(samples, test_size=0.3)
correction=0.2
# We define a function generator which takes input at the samples, in a batch size of 32


def generator(samples, batch_size):
    num_samples = len(samples)
    # We use while 1, so that while loop, loop's forever. We donot want the generator to terminate
    while True: 
        # Shuffle samples, changes the sequence of samples, the purpose is to avoid homogeniety of samples
        shuffle(samples)
        # We make batches and then deal with each individual batch during each iteration of the for loop
        for offset in range(0, num_samples, batch_size):
            # Below batch_sample has the 1st batch in the first iteration
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            # For this batch, we read in images and steering angles for each element of the batch
            for batch_sample in batch_samples:
                source_path=batch_sample[0]
                filename=source_path.split('/')[-1]
                current_path=image_datapath+filename
#                 image_BGR=cv2.imread(current_path)
#                 image=cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)

                # image = ndimage.imread(current_path)
                images.append(image)
                # Flipping the image taken from the center camera
                image_flip=cv2.flip(image,1) 
                images.append(image_flip)
                angle=float(batch_sample[3])
                angles.append(angle)
                # Flipping the angle for the corresponding center camera image
                measure_flip=float(angle*-1.0)  
                angles.append(measure_flip)
                # Images with Minus brightness
                image_bright = color.rgb2hsv(image)
                image_bright[:, :, 2] *= .5 + .4 * np.random.uniform()
                image_bright = img_as_ubyte(color.hsv2rgb(image_bright))
                images.append(image_bright)
                measure_bright = angle
                angles.append(measure_bright)
                # Images with Equalize histogram
                image_hist = np.copy(image)
                for channel in range(image_hist.shape[2]):
                    image_hist[:, :, channel] = exposure.equalize_hist(image_hist[:, :, channel]) * 255
                images.append(image_hist)
                measure_hist = angle
                angles.append(measure_hist)
                # Images that are Blurred
                image_blur = img_as_ubyte(np.clip(filters.gaussian(image, multichannel=True), -1, 1))
                images.append(image_blur)
                measure_blur = angle
                angles.append(measure_blur)
                # Image that are Noisy
                image_noise = img_as_ubyte(random_noise(image, mode='gaussian'))
                images.append(image_noise)
                measure_noise = angle
                angles.append(measure_noise)
              
            for batch_sample in batch_samples:
                source_path=batch_sample[1]
                filename=source_path.split('/')[-1]
                current_path=image_datapath+filename
                image_BGR=cv2.imread(current_path)
                image=cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
                # image = ndimage.imread(current_path)
                images.append(image)
                # Angle correction for the image taken from the left side camera
                angle=float(batch_sample[3])+correction 
                angles.append(angle)
                
            for batch_sample in batch_samples:
                source_path=batch_sample[2]
                filename=source_path.split('/')[-1]
                current_path=image_datapath+filename
                image_BGR=cv2.imread(current_path)
                image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
                # image = ndimage.imread(current_path)
                images.append(image)
                # Angle correction for the image taken from the right side camera
                angle=float(batch_sample[3])-correction 
                angles.append(angle)
                
            X_train=np.array(images)
            y_train=np.array(angles)
            # Shuffling the data
            yield sklearn.utils.shuffle(X_train,y_train)   

# Here we use the generator funtion to get the trainning and validation data 
# Later we will use this data to compile and train the model 
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

print('Importing keras libraries')

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt

# In this project we are using the CNN architecture by NVIDIA, ref: https://arxiv.org/pdf/1604.07316v1.pdf
model = Sequential()
# The first step before trainning is to preprocess incoming data. 
# For this purpose we will do normalisation for each model, That is to say that the data should be centered around zero with small standard deviation 

# Model 1 is the LeNet Model
def Model1():
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=Img_shape))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    #Layer 1- Convolution, no of filters- 6, filter size= 5x5, stride= 1x1, activation = Rectified linear unit
    model.add(Conv2D(6,(5,5),activation='relu'))
    model.add(MaxPooling2D())
    #Layer 2- Convolution, no of filters- 6, filter size= 5x5, stride= 1x1, activation = Rectified linear unit
    model.add(Conv2D(6,(5,5),activation='relu'))
    model.add(MaxPooling2D())
    #Flatten image from 2D to side by side
    model.add(Flatten())
    #Layer 3- Fully connected layer 1
    model.add(Dense(120))
    #Layer 4- Fully connected layer 2
    model.add(Dense(84))
    #Layer 5- Fully connected layer 3
    model.add(Dense(1))
    return model

# Model 2 is the Nvidia Model
def Model2():
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=Img_shape))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    #Layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2, activation = Rectified linear unit
    model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu'))
    #Layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2, activation = Rectified linear unit
    model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu'))
    #Layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2, activation = Rectified linear unit
    model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu'))
    #Layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1, activation = Rectified linear unit
    model.add(Conv2D(64,(3,3), activation='relu'))
    #Layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1, activation = Rectified linear unit
    model.add(Conv2D(64,(3,3), activation='relu'))
    #Flatten image from 2D to side by side
    model.add(Flatten())
    #Layer 6- Fully connected layer 1 with activation = Rectified linear unit
    model.add(Dense(100))
    #Layer 7- Fully connected layer 2 with activation = Rectified linear unit
    model.add(Dense(50))
    #Layer 8- Fully connected layer 3 with activation = Rectified linear unit
    model.add(Dense(10))
    #Layer 9- Fully connected layer 4 
    model.add(Dense(1))
    return model

# Model 3 is the modified Nvidia Model, with Exponential linear unit and a Drop out layer.
def Model3():
  
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=Img_shape))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    #Layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2, activation = exponential linear unit
    model.add(Conv2D(24,(5,5), strides=(2,2), activation='elu'))
    #Layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2, activation = exponential linear unit
    model.add(Conv2D(36,(5,5), strides=(2,2), activation='elu'))
    #Layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2, activation = exponential linear unit
    model.add(Conv2D(48,(5,5), strides=(2,2), activation='elu'))
    #Layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1, activation = exponential linear unit
    model.add(Conv2D(64,(3,3), activation='elu'))
    #Layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1, activation = exponential linear unit
    model.add(Conv2D(64,(3,3), activation='elu'))
    #Flatten image from 2D to side by side
    model.add(Flatten())
    #Layer 6- Fully connected layer 1 with activation = exponential linear unit
    model.add(Dense(100, activation='elu'))
    #Adding a dropout layer to avoid overfitting. Here we are have given the dropout rate as 25% after first fully connected layer
    model.add(Dropout(0.25))
    #Layer 7- Fully connected layer 2 with activation = exponential linear unit
    model.add(Dense(50, activation='elu'))
    #Layer 8- Fully connected layer 3 with activation = exponential linear unit
    model.add(Dense(10, activation='elu'))
    #Layer 9- Fully connected layer 4 
    model.add(Dense(1)) 
    #Final layer has contained one value as this is a regression problem and not classification
    return model

# Model 4 is the VGG16Model
def Model4():
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=Img_shape)) 
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Conv2D(32,(3,3), activation='relu', padding='same'))
    model.add(Conv2D(32,(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Conv2D(32,(3,3), activation='relu', padding='same'))
    model.add(Conv2D(32,(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))              
    return model

model = Model3()

print('Training...')
# print(model.summary())

# # Checkpoint to save model 
# checkpoint = ModelCheckpoint("model3.h5", monitor="val_loss", mode="min", save_best_only = True, verbose=1)

# # For early stopping : If the val_loss is not improving
# earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0.0001, patience = 2, verbose = 1, restore_best_weights = True)


# callbacks_list = [earlystop, checkpoint]

steps_per_epoch = int(len(train_samples)/32)-1
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples),
                    verbose=1,epochs=1)
#                    callbacks=callbacks_list,
                    
# history_object =model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,   nb_val_samples=len(validation_samples), nb_epoch=1, verbose=1)


model.save('model3.h5')
# Print the keys contained in the history object
print(history_object.history.keys())
# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# python drive.py model.h5 run1
# In order to visualize the model architecture
# plot_model(model, to_file='ModelPlot.png', show_shapes=True, show_layer_names=True)
# For reducing the Learning Rate : if the val_loss remains constant or is not improving
# reducelearnrate = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 2, verbose = 1, min_delta = 0.0001)
# , reducelearnrate