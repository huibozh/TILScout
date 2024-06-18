# -*- coding: utf-8 -*-
"""
Autoencoder k means

@author: Huibo Zhang
"""
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.cluster import KMeans
from pandas.core.frame import DataFrame
from tensorflow.keras.callbacks import EarlyStopping


# encoder
def build_encoder(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    return Model(input_layer, encoded, name="encoder")

# decoder
def build_decoder(encoded_shape):
    encoded_input = Input(shape=encoded_shape)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    return Model(encoded_input, decoded, name="decoder")


encoder = build_encoder((152, 152, 3))
decoder = build_decoder(encoder.output_shape[1:])
autoencoder = Model(encoder.input, decoder(encoder.output))
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
autoencoder.summary()

# Image data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'Train_new/train/',
    target_size=(152, 152),
    batch_size=32,
    class_mode=None,
    shuffle=False)

validation_generator = validation_datagen.flow_from_directory(
    'Train_new/test/',
    target_size=(152, 152),
    batch_size=32,
    class_mode=None,
    shuffle=False)

def autoencoder_generator(image_data_generator):
    for img_data in image_data_generator:
        yield img_data, img_data  

train_autoencoder_generator = autoencoder_generator(train_generator)
validation_autoencoder_generator = autoencoder_generator(validation_generator)

checkpoint_path = 'autoencoder_model_new.h5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', save_best_only=True, mode='min', verbose=1)

early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=5,         
    restore_best_weights=True, 
    verbose=1
)


autoencoder.fit(
    train_autoencoder_generator,
    epochs=50,  
    steps_per_epoch=len(train_generator),
    validation_data=validation_autoencoder_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint, early_stopping]
)


model = load_model(checkpoint_path)

X_train_encoded = encoder.predict(train_generator)
X_test_encoded = encoder.predict(validation_generator)

X_train_encoded_reshape = X_train_encoded.reshape(len(X_train_encoded), -1)
X_test_encoded_reshape = X_test_encoded.reshape(len(X_test_encoded), -1)

kmeans = KMeans(n_clusters=3, random_state=42)
y_train_pred = kmeans.fit_predict(X_train_encoded_reshape)


label_map = train_generator.class_indices
labels = []
for class_name, class_index in label_map.items():
    class_dir = os.path.join('Train_new/train/', class_name)
    num_images = len(os.listdir(class_dir))
    labels += [class_index] * num_images

Y_train = np.array(labels)

## accuracy score 
Y_train_pred = DataFrame(y_train_pred)
Y_train_pred.index.name="No"
Y_label = DataFrame(Y_train)
Y_label.index.name="No"
pred_results=pd.merge(Y_train_pred,Y_label,on='No')
pred_results.columns=["prediction","label"]
pred_results.to_csv("plot/Autoencoder_Kmeans_train.csv")


"""
####### test set
"""

y_test_pred = kmeans.predict(X_test_encoded_reshape)

label_map2 = validation_generator.class_indices
labels2 = []
for class_name, class_index in label_map2.items():
    class_dir = os.path.join('Train_new/test/', class_name)
    num_images = len(os.listdir(class_dir))
    labels2 += [class_index] * num_images
Y_test = np.array(labels2)

##accuracy score 
Y_test_pred = DataFrame(y_test_pred)
Y_test_pred.index.name="No"
Y_label = DataFrame(Y_test)
Y_label.index.name="No"
pred_results=pd.merge(Y_test_pred,Y_label,on='No')
pred_results.columns=["prediction","label"]
pred_results.to_csv("plot/Autoencoder_Kmeans_test_results.csv")

