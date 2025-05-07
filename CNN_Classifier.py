
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

#Fetching MNIST dataset from tensorflow dataset
mnist_data= tf.keras.datasets.mnist

#Spliting dataset b/w train and test set
(x_train, labels_train),(x_test, labels_test) = mnist_data.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#Normalizing the data
x_train /= 255
x_test /= 255

from keras.utils import to_categorical

# Convert the output layer to provide the result in one-hot encoding
y_train = to_categorical(labels_train, 10)
y_test = to_categorical(labels_test, 10)

#Reshaping the data to 4D array
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

import numpy as np

# Generate new data

# specify the arguments
rotation_range_val = 15
width_shift_val = 0.1
height_shift_val = 0.1
shear_range_val= 25
zoom_range_val=[0.7,1.3]

# import relevant library
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# create the class object
datagen_train = ImageDataGenerator(rotation_range = rotation_range_val,
                             width_shift_range = width_shift_val,
                             height_shift_range = height_shift_val,
                             zoom_range=zoom_range_val)

sample_x_train = x_train.copy()
sample_labels_train = labels_train.copy()

# fit the generator
datagen_train.fit(sample_x_train.reshape(sample_x_train.shape[0], 28, 28, 1))

# Total number of test data
num = 12000

# Generate new augmented data
for new_x_train, new_y_train in datagen_train.flow(sample_x_train.reshape(sample_x_train.shape[0], 28, 28, 1),sample_labels_train.reshape(sample_labels_train.shape[0], 1),batch_size=num,shuffle=False):
    break

#Flattening the new data
new_y_train = new_y_train.flatten()
new_y_train = to_categorical(new_y_train, 10)

#Concatenating the new data to the original data
x_train = np.concatenate((x_train, new_x_train), axis=0)
y_train = np.concatenate((y_train, new_y_train), axis=0)

print("X train: ", x_train.shape)
print("Y train: ", y_train.shape)
print("X_test: ", x_test.shape)
print("Y_test: ", y_test.shape)

"""M7 Model"""

#with strategy.scope():
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input, BatchNormalization, Concatenate, Average, Maximum,InputLayer
from keras.models import Model
from keras import initializers, optimizers
model_M7 =Sequential()#Initialising sequential model
model_M7.add(InputLayer((28, 28,1)))#Making input layer.
model_M7.add(Conv2D(filters =32,kernel_size=(3,3),activation="relu" ,name="layer1"))#Making 1st Conv2d layer with relu activation.
model_M7.add(BatchNormalization())#Making first BatchNorm2d layer .like wise all below layers.
model_M7.add(MaxPool2D(2,2))
model_M7.add(Dropout(0.25))
model_M7.add(Conv2D(filters =32,kernel_size=(3,3),activation="relu" ,name="layer2"))
model_M7.add(BatchNormalization())
model_M7.add(MaxPool2D(2,2))
model_M7.add(Dropout(0.25))
model_M7.add(Conv2D(filters =64,kernel_size=(3,3),activation="relu",name="layer3" ))
model_M7.add(BatchNormalization())
model_M7.add(Dropout(0.25))
model_M7.add(Conv2D(filters =128,kernel_size=(3,3),activation="relu" ,name="layer4"))
model_M7.add(BatchNormalization())
model_M7.add(Dropout(0.25))
model_M7.add(Dense(10))#Making Linear /last layer .
model_M7.add(BatchNormalization())
model_M7.add(Flatten())#Flattening output to compare with labels.
model_M7.add(Dense(10,activation='softmax'))



model_M7.compile(optimizer=optimizers.Adam(learning_rate=0.001,beta_1=0.98)
            ,loss='categorical_crossentropy',
            metrics=['accuracy']) #Optimization
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
reduceLR = ReduceLROnPlateau(monitor='val_accuracy', factor=0.75, patience=3, min_lr=0.0001)
early_stopping_callback = EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy')
filepath="model_M7-rot14-wid0.1-hei0.1-zoom0.7to1.3/{epoch:02d}-{val_accuracy:.5f}.h5"
checkpoint = ModelCheckpoint(filepath,monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

z_train=model_M7.fit(x_train,y_train,epochs=100,callbacks=[reduceLR, early_stopping_callback,checkpoint], validation_data=(x_test, y_test)) #Model Training. You can try epochs=150 too.

# Generate new data for testing

# specify the arguments
rotation_range_val = 10
width_shift_val = 0.1
height_shift_val = 0.1
shear_range_val= 25
zoom_range_val=[1.0,1.5]

# import relevant library
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# create the class object
datagen = ImageDataGenerator(rotation_range = rotation_range_val,
                            width_shift_range = width_shift_val,
                            height_shift_range = height_shift_val,
                            zoom_range=zoom_range_val)

sample_x_test = x_test.copy()
sample_y_test = labels_test.copy()

# fit the generator
datagen.fit(sample_x_test.reshape(sample_x_test.shape[0], 28, 28, 1))

# Total number of test data
num = 10000

# Generate new augmented data
for new_x_test, new_y_test in datagen.flow(sample_x_test.reshape(sample_x_test.shape[0], 28, 28, 1),sample_y_test.reshape(sample_y_test.shape[0], 1),batch_size=num,shuffle=False):
    break

#Flattening the new test data
new_y_test = new_y_test.flatten()
new_y_test = to_categorical(new_y_test, 10)

#Concatenating the new test data to the original test data
x_test = np.concatenate((x_test, new_x_test), axis=0)
y_test = np.concatenate((y_test, new_y_test), axis=0)

print("Updated x_test shape: ",x_test.shape)
print("Updated y_test shape: ",y_test.shape)

model_M7.summary()

z_test=model_M7.evaluate(x_test,y_test) #Model evaluation on Test data

z_train.history.keys()

training_accuracy=z_train.history["accuracy"] #Accuracy on training data
plt.plot(training_accuracy) #ploting curve
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.title("M5 model Evaluation")
plt.legend(["Train"])
plt.show()

# Testing the model on the nearly mnist dataset

# Get the nearly mnist from Kaggle to test
import numpy as np

data = np.genfromtxt('nearly_mnist.csv', delimiter=',', skip_header=1, invalid_raise = False)
target_flat = data[:,-1]
data = data[:,:-1]
print(data.shape)
print(target_flat.shape)
data = data.astype('float32')
data /= 255
target = keras.utils.to_categorical(target_flat, 10)
data = data.reshape(data.shape[0], 28, 28, 1)
print(data.shape)
print(target.shape)

# Dynamically finding the best model to be used.
import os
import re

# Name for the directory where the models are stored
model_dir = 'model_M7-rot14-wid0.1-hei0.1-zoom0.7to1.3'

# List all .h5 files in the directory
model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]

# Function to extract accuracy from filename
def extract_accuracy(filename):
    match = re.search(r'-(\d+\.\d+)\.h5$', filename)
    return float(match.group(1)) if match else 0.0

# Find the file with the highest accuracy
best_model_file = max(model_files, key=extract_accuracy)
best_model_path = os.path.join(model_dir, best_model_file)

print(f"Loading best model: {best_model_path}")
model = keras.models.load_model(best_model_path)
outputs = model.predict(data)
labels_predicted = np.argmax(outputs, axis=1)
correct_classified = sum(labels_predicted == target_flat)
print('Percentage correctly classified MNIST=', 100*correct_classified/target_flat.size)