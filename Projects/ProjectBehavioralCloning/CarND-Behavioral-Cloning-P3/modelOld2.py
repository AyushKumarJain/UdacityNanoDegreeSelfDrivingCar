import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# Code to retrieve driving data from driving_log.csv file
samples = []
with open('/home/workspace/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # Line per line reading the data and appending to samples list
    for line in reader:
        samples.append(line)
                   
# If there is header in csv data, we donot want it to be uploaded into samples list, so we will use samples = samples[1:],
samples=samples[1:]   
print('Filenames loaded...')

image_datapath = "/home/workspace/CarND-Behavioral-Cloning-P3/data/IMG/"

# Splitting the data saved in samples list into training and testing data. It is 80-20 split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# We define a function generator which takes input at the samples, in a batch size of 32
def generator(samples, batch_size):
    num_samples = len(samples)
    # We use while 1, so that while loop, loop's forever. We donot want the generator to terminate
    while 1: 
        # Shuffle samples, changes the sequence of samples, the purpose is to avoid homogeniety of samples
        shuffle(samples)
        # We make batches and then deal with each individual batch during each iteration of the for loop
        for offset in range(0, num_samples, batch_size):
            # Below batch_sample has the 1st batch in the first iteration
            batch_samples = samples[offset:offset+batch_size]
            
            # Defining images and angle list to save the 3 images and 3 angles after modification
            images = []
            angles = []
            # For this batch, we read in images and steering angles for each element of the batch
            for batch_sample in batch_samples:
                # Each batch_sample has 3 images, that is from the center, left and right camera.
                # We retrive them in 3 variables
                center_image = cv2.imread((image_datapath+batch_sample[0]).replace(' ',''))
                left_image = cv2.imread((image_datapath+batch_sample[1]).replace(' ',''))
                right_image = cv2.imread((image_datapath+batch_sample[2]).replace(' ',''))

                # We then retrieve the steering angle corresponding to the center image.
                # Using this steering angle we apply correction to the right and left image steering angle
                center_angle = float(batch_sample[3])
                correction = 0.2
                # We need to do this correction, if we treat left images as center
                left_angle = center_angle + correction 
                # We need to do this correction, if we treat right images as center
                right_angle = center_angle - correction 
                # Cropping images, as the top of the image is not usable in detecting the lane
                # Saving the cropped images and the corresponding angles in the images and angles list
                images.extend([center_image[70:135, :],left_image[70:135, :],right_image[70:135, :]]) 
                angles.extend([center_angle,left_angle,right_angle])
                
                # We do this for all the elemenets on the current batch.
            
            # For this batch we also want to augment the data set. 
            # The first step that we do to augment the data set is to flip the image, so that the bias to more only left or only right is removed.
            # Defininf the aumented_images and augmented_angle data set
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1)) # flipping image for data augmentation
                augmented_angles.append(angle * -1.0)
            # Finally we get the training input and output data set corresponding to this batch.
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            
            # Again we shuffle the data for the first batch. 
            yield sklearn.utils.shuffle(X_train, y_train)
            # Finally when this loop runs over all the batches. We get a trainning data set with images as inputs and output labels as steering angle

# Here we use the generator funtion to get the trainning and validation data 
# Later we will use this data to compile and train the model 
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

row, col, ch = 65, 320, 3

print('Importing keras libraries')

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
import matplotlib.pyplot as plt

# In this project we are using the CNN architecture by NVIDIA, ref: https://arxiv.org/pdf/1604.07316v1.pdf
model = Sequential()
# The first step before trainning is to preprocess incoming data. 
# For this purpose we will do normalisation for each model, That is to say that the data should be centered around zero with small standard deviation 

# Model 1 is the LeNet Model
def Model1():
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/127.5 - 1, input_shape=(row, col, ch)))
    #Layer 1- Convolution, no of filters- 6, filter size= 5x5, stride= 1x1, activation = Rectified linear unit
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    #Layer 2- Convolution, no of filters- 6, filter size= 5x5, stride= 1x1, activation = Rectified linear unit
    model.add(Convolution2D(6,5,5,activation='relu'))
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
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch))) 
    #Layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2, activation = Rectified linear unit
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    #Layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2, activation = Rectified linear unit
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    #Layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2, activation = Rectified linear unit
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    #Layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1, activation = Rectified linear unit
    model.add(Convolution2D(64,3,3, activation='relu'))
    #Layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1, activation = Rectified linear unit
    model.add(Convolution2D(64,3,3, activation='relu'))
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
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/127.5 - 1, input_shape=(row, col, ch))) 
    #Layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2, activation = exponential linear unit
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='elu'))
    #Layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2, activation = exponential linear unit
    model.add(Convolution2D(36,5,5,subsample=(2,2), activation='elu'))
    #Layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2, activation = exponential linear unit
    model.add(Convolution2D(48,5,5,subsample=(2,2), activation='elu'))
    #Layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1, activation = exponential linear unit
    model.add(Convolution2D(64,3,3, activation='elu'))
    #Layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1, activation = exponential linear unit
    model.add(Convolution2D(64,3,3, activation='elu'))
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

model = Model2()

print('Training...')

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples*6), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

# Note, the total training samples are 6 times per epoch counting both original
# and flipped left, right and center images
model.save('model2.h5')
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

