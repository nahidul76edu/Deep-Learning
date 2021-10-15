#Import Necessary Libraries
import keras
from keras.layers import Input, Lambda, Dense, Flatten, merge
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation
from keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

####################################################################
#Import your Dataset
Left = np.load('filename.npy', allow_pickle=True)
x_Left=np.array([i[0] for i in Left])
y_Left=np.array([i[1] for i in Left])
x_Left=x_Left/255

Right = np.load('filename.npy', allow_pickle=True)
x_Right=np.array([i[0] for i in Right])
y_Right=np.array([i[1] for i in Right])
x_Right=x_Right/255

#######################################################################
#concatenate the data and labels 
X = np.concatenate((x_Left, x_Right)) #all data
Y = np.concatenate((y_Left, y_Right)) #all labels

#######################################################################
# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0)

#########################################################################
num_classes = 2 #number of class or labels
image_input = Input(shape=(224, 224, 3))
model = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')
#model.summary()
last_layer = model.get_layer('block5_conv1').output
x= Flatten(name='flatten')(last_layer)
x = Dense(1024, activation='relu', name='fc1')(x)
x = Dropout(0.2)(x) 
x = Dense(512, activation='relu', name='fc2')(x)
x = Dropout(0.4)(x) 
x = Dense(288, activation='relu', name='fc3')(x)
x = Dropout(0.6)(x)
out = Dense(num_classes, activation='softmax', name='output')(x)
custom_vgg_model = Model(image_input, out)

##########################################################################
#pre-trained weight transfer 
#here I used a total of 11 layers where the pre-trained will be transferred, In your model, you can set the layers as you want. 
#Rest of the layer's weights will be updated according to your dataset.

for layer in custom_vgg_model.layers[:11]:
  layer.trainable = False
  
#To see your models layout.
custom_vgg_model.summary()

##########################################################################
#Compile and fit the architecture with your dataset.
custom_vgg_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = custom_vgg_model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), batch_size = 64)