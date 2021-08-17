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
hari


my_and_udacity_data = ['./data'] 
               
samples = []
# with open('/home/workspace/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
for list in [my_and_udacity_data]:
    for filename in list:
        with open(filename + '/driving_log.csv') as csvfile:mo nmo 
            for line in csv.reader(csvfile):
                samples.append(line)mo naray
                   
samples=samples[1:]   
print('Filenames loaded...')
print(len(samples))
image_datapath = "./data/IMG/"
row, col, ch = 160, 320, 3
Img_shape= (row, col, ch)
correction=0.2
batchsize=32
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# print(len(samples))
print(len(train_samples))
print(len(validation_samples))


def generator(samples, batch_size):
    num_samples = len(samples)
    while True: 
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                source_path=batch_sample[0]
                filename=source_path.split('/')[-1]
                current_path=image_datapath+filename
                image=cv2.imread(current_path)
                images.append(image)
                angle=float(batch_sample[3])
                angles.append(angle)
                              
                # Flipped images 
                image_flip=cv2.flip(image,1) 
                images.append(image_flip)
                measure_flip=float(angle*-1.0)  
                angles.append(measure_flip)
                
                # Right images corrected         
                source_path=batch_sample[1]
                filename=source_path.split('/')[-1]
                current_path=image_datapath+filename
                image=cv2.imread(current_path)
                images.append(image)
                angle=float(batch_sample[3])+correction 
                angles.append(angle)

                # Left images corrected
                source_path=batch_sample[2]
                filename=source_path.split('/')[-1]
                current_path=image_datapath+filename
                image=cv2.imread(current_path)
                images.append(image)
                angle=float(batch_sample[3])-correction 
                angles.append(angle)


            X_train=np.array(images)
            y_train=np.array(angles)
            yield sklearn.utils.shuffle(X_train,y_train)   

train_generator = generator(train_samples, batch_size=batchsize)
validation_generator = generator(validation_samples, batch_size=batchsize)

print('Importing keras libraries')

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt

model = Sequential()

# Model 1: LeNet Model
def Model1():
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=Img_shape))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Conv2D(6,(5,5),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(6,(5,5),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

# Model 2: Nvidia Model
def Model2():
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=Img_shape))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

# Model 3: modified Nvidia Model
def Model3():
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=Img_shape))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Conv2D(24,(5,5), strides=(2,2), activation='elu'))
    model.add(Conv2D(36,(5,5), strides=(2,2), activation='elu'))
    model.add(Conv2D(48,(5,5), strides=(2,2), activation='elu'))
    model.add(Conv2D(64,(3,3), activation='elu'))
    model.add(Conv2D(64,(3,3), activation='elu'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.25))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1)) 
    return model

# Model 4: VGG16Model
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

checkpoint = ModelCheckpoint("model1.h5", monitor="val_loss", mode="min", save_best_only = True, verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0.0001, patience = 2, verbose = 1, restore_best_weights = True)

reducelearnrate = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 2, verbose = 1, min_delta = 0.0001)

callbacks_list = [earlystop, checkpoint, reducelearnrate]

steps_per_epoch = int(len(train_samples*4)/batchsize)
validation_steps_per_epoch = int(len(validation_samples*4)/batchsize)

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation_generator,
                    validation_steps=validation_steps_per_epoch,
                    verbose=1,epochs=7,
                    callbacks=callbacks_list)
                    

model.save('model1.h5')
print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()




















# python drive.py model.h5 run1

# plot_model(model, to_file='ModelPlot.png', show_shapes=True, show_layer_names=True)

# reducelearnrate = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 2, verbose = 1, min_delta = 0.0001)

#                 # Image Noisy
#                 image_noise = img_as_ubyte(random_noise(image, mode='gaussian'))
#                 images.append(image_noise)
#                 measure_noise = angle
#                 angles.append(measure_noise)

#                 # Images Blurred
#                 image_blur = img_as_ubyte(np.clip(filters.gaussian(image, multichannel=True), -1, 1))
#                 images.append(image_blur)
#                 measure_blur = angle
#                 angles.append(measure_blur)
                
                
#                 # Flipped Images with Minus brightness
#                 image_bright = color.rgb2hsv(image_flip)
#                 image_bright[:, :, 2] *= .5 + .4 * np.random.uniform()
#                 image_bright = img_as_ubyte(color.hsv2rgb(image_bright))
#                 images.append(image_bright)
#                 measure_bright = measure_flip
#                 angles.append(measure_bright)
                
#                 # Flipped Images with Equalize histogram
#                 image_hist = np.copy(image_flip)
#                 for channel in range(image_hist.shape[2]):
#                     image_hist[:, :, channel] = exposure.equalize_hist(image_hist[:, :, channel]) * 255
#                 images.append(image_hist)
#                 measure_hist = measure_flip
#                 angles.append(measure_hist)