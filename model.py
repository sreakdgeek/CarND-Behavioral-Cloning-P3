
# coding: utf-8

# ## Behavior Cloning Project

# In[1]:


import os
import csv

from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda, Dense, Conv2D, Activation, Flatten, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from random import shuffle
import cv2
import numpy as np
import sklearn


# ## Read Driving log

# In[2]:


samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


# ## Train - Validation split

# In[3]:


import matplotlib.pyplot as plt
import random


retain_probability = 0.33

# Plot histogram - before dropping zero angles
center_angles_before = np.array([item[3] for item in samples]).astype(float)

plt.xlabel('Center Angles')
plt.ylabel('Distribution')
plt.title('Distribution before dropping zero angles')
plt.hist(center_angles_before)
plt.show()

print("Length of samples = " + str(len(samples)))

# Eliminate 67% of the zero angles
samples_new = [item for item in samples if ((random.random() <= retain_probability and float(item[3]) == 0) 
                                            or float(item[3]) != 0)]

print("Length of samples new = " + str(len(samples_new)))

center_angles_after = np.array([item[3] for item in samples_new]).astype(float)

plt.xlabel('Center Angles')
plt.ylabel('Distribution')
plt.title('Distribution after dropping zero angles')
plt.hist(center_angles_after)
plt.show()

train_samples, validation_samples = train_test_split(samples_new, test_size=0.2)

print("Number of training samples = " + str(len(train_samples)))
print("Number of validation samples = " + str(len(validation_samples)))


# ## Generator will stream training sampeles while training

# In[4]:


import matplotlib.pyplot as plt
import matplotlib

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.2
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                # Center
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2YUV)
                
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                # Left
                name = './IMG/' + batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2YUV)
                images.append(left_image)
                
                # Left correction
                left_angle = center_angle + correction
                angles.append(left_angle)
               
                # Right
                name = './IMG/' + batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2YUV)
                images.append(right_image)
                
                # Right correction
                right_angle = center_angle - correction
                angles.append(right_angle)
                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)


# In[5]:


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# ## Model Architecture

# In[ ]:


from keras.callbacks import ModelCheckpoint
import keras
from keras import regularizers
from keras import optimizers
from keras import backend as K


ch, row, col = 3, 80, 320  # Trimmed image format
batch_size = 32

# set up cropping2D layer
model = Sequential()

# Layer 1 - Cropping Layer

model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))

# Preprocess incoming data, centered around zero with small standard deviation 

# Layer 2 - Pre-processing Layer - Normalization
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))

# Layer 3 - Conv 1
model.add(Conv2D(32, (5,5), padding = 'same'))
model.add(Activation('relu'))

# Layer 4 - Conv 2
model.add(Conv2D(32, (5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Layer 5 - Conv 3
model.add(Conv2D(48, (5,5), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Layer 6 - Conv 4
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Layer 7 - Conv 5
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Layer 8 - Flatten
model.add(Flatten())

# Layer 9 - FC - 1
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.3))

# Layer 10 - FC - 2
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Layer 11 - FC - 3
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
          
# Layer 12 - Dense final output layer
model.add(Dense(1))

#adam_opt = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#model.compile(loss='mse', optimizer=adam_opt)

model.compile(loss='mse', optimizer='adam')

filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
callbacks_list = [checkpoint]


history_object = model.fit_generator(train_generator, 
                    steps_per_epoch = len(train_samples) // batch_size, 
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples) // batch_size,
                    callbacks = callbacks_list,
                    epochs=12)


# In[ ]:


from keras.callbacks import ModelCheckpoint
import keras
from keras import regularizers


ch, row, col = 3, 80, 320  # Trimmed image format
batch_size = 32

# set up cropping2D layer
model = Sequential()

# Layer 1 - Cropping Layer

model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))

# Preprocess incoming data, centered around zero with small standard deviation 

# Layer 2 - Pre-processing Layer
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))

# Layer 3 - Conv 1
model.add(Conv2D(32, (5,5), padding = 'same'))
model.add(keras.layers.ELU(alpha=1.0))

# Layer 4 - Conv 2
model.add(Conv2D(32, (5,5), kernel_regularizer=regularizers.l2(0.01)))
model.add(keras.layers.ELU(alpha=1.0))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 5 - Conv 3
model.add(Conv2D(64, (5,5), padding = 'same', kernel_regularizer=regularizers.l2(0.01)))
model.add(keras.layers.ELU(alpha=1.0))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 6 - Conv 4
model.add(Conv2D(64, (3,3), kernel_regularizer=regularizers.l2(0.01)))
model.add(keras.layers.ELU(alpha=1.0))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 7 - Conv 5
model.add(Conv2D(64, (3,3), kernel_regularizer=regularizers.l2(0.01)))
model.add(keras.layers.ELU(alpha=1.0))
model.add(MaxPooling2D(pool_size=(2,2)))
          
# Layer 8 - Flatten
model.add(Flatten())

# Layer 9 - FC - 1
model.add(Dense(512))
model.add(keras.layers.ELU(alpha=1.0))

# Layer 10 - FC - 2
model.add(Dense(256))
model.add(keras.layers.ELU(alpha=1.0))

# Layer 11 - FC - 3
model.add(Dense(128))
model.add(keras.layers.ELU(alpha=1.0))

# Layer 12 - Dense final output layer
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
callbacks_list = [checkpoint]


history_object = model.fit_generator(train_generator, 
                    steps_per_epoch = len(train_samples) // batch_size, 
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples) // batch_size,
                    callbacks = callbacks_list,
                    epochs=12)


# ## Training vs Validation Loss

# In[ ]:


model.save('model.h5')


# In[8]:


get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


# In[ ]:


get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

