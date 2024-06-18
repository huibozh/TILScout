# -*- coding: utf-8 -*-
"""
InceptionV3
@author: Huibo Zhang
"""

import numpy as np 
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

InceptionV3_model = Sequential()
pretrained_model= tf.keras.applications.InceptionV3(include_top=False,
                   input_shape=(224,224,3),
                   pooling='avg',classes=3,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False
InceptionV3_model.add(pretrained_model)
InceptionV3_model.add(Flatten())
InceptionV3_model.add(Dense(512, activation='relu'))
InceptionV3_model.add(Dense(3, activation='softmax'))
InceptionV3_model.summary()


"""
######### input data
"""
# ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
    'Train_new/train/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    shuffle=False)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    'Train_new/test/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    shuffle=False)

####training
InceptionV3_model.compile(optimizer='adam', loss = "sparse_categorical_crossentropy",metrics = ['accuracy'])
checkpointer = tf.keras.callbacks.ModelCheckpoint('InceptionV3_model.h5', verbose=1, save_best_only=True)
callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, monitor='loss',mode="auto"),
        tf.keras.callbacks.TensorBoard(log_dir='logs'),
        checkpointer]

history = InceptionV3_model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    verbose=1,
    callbacks=callbacks)


"""
### accuracy and loss plot
"""
accu = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accu))

plt.plot(epochs,accu, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='validation accuracy')
plt.title('Training and validation set accuracy')
plt.legend(loc='lower right')
plt.savefig('./plot/InceptionV3_accuracy.pdf', dpi = 1000)
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('./plot/InceptionV3_loss.pdf', dpi = 1000)
plt.show()


"""
######### prediction
"""
InceptionV3_model = keras.models.load_model('InceptionV3_model.h5')
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score, roc_auc_score#, confusion_matrix
from sklearn.preprocessing import label_binarize

def predict_model(model, generator, steps):
    generator.reset()  
    pred = model.predict(generator, steps=steps, verbose=1)
    return pred

def calculate_metrics(y_true, y_pred, y_pred_proba, num_classes=3):
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
        
    y_true_binarized = label_binarize(y_true, classes=range(num_classes))
    auc = roc_auc_score(y_true_binarized, y_pred_proba, multi_class='ovr', average='weighted')
    
    return accuracy, kappa, f1, precision, recall, auc

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'Train_new/train/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse',
        shuffle=False)

validation_generator = validation_datagen.flow_from_directory(
        'Train_new/test/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse',
        shuffle=False)

train_steps = np.ceil(train_generator.samples / train_generator.batch_size)
val_steps = np.ceil(validation_generator.samples / validation_generator.batch_size)

train_pred_proba = predict_model(InceptionV3_model, train_generator, steps=train_steps)
val_pred_proba = predict_model(InceptionV3_model, validation_generator, steps=val_steps)

train_pred_labels = np.argmax(train_pred_proba, axis=1)
val_pred_labels = np.argmax(val_pred_proba, axis=1)

train_true_labels = train_generator.classes
val_true_labels = validation_generator.classes

train_metrics = calculate_metrics(train_true_labels, train_pred_labels, train_pred_proba, num_classes=3)
val_metrics = calculate_metrics(val_true_labels, val_pred_labels, val_pred_proba, num_classes=3)

metrics_names = ['Accuracy', 'Kappa', 'F1', 'Precision', 'Recall', 'AUC']
for name, train_metric, val_metric in zip(metrics_names, train_metrics, val_metrics):
    print(f"{name} - Train: {train_metric}, Validation: {val_metric}")

# specificity
def calculate_specificity(y_true, y_pred, num_classes):
    specificity_scores = []
    for i in range(num_classes):
        true_negatives = np.sum((y_true != i) & (y_pred != i))
        false_positives = np.sum((y_true != i) & (y_pred == i))
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        specificity_scores.append(specificity)
    return specificity_scores

# AUC
def calculate_auc(y_true, y_score, num_classes):
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    auc_scores = []
    for i in range(num_classes):
        if y_true_bin[:, i].sum() > 0:  
            auc_score = roc_auc_score(y_true_bin[:, i], y_score[:, i])
            auc_scores.append(auc_score)
        else:
            auc_scores.append(float('nan'))  
    return auc_scores

num_classes = 3  
# training set
train_precision = precision_score(train_true_labels, train_pred_labels, average=None)
train_recall = recall_score(train_true_labels, train_pred_labels, average=None)
train_f1 = f1_score(train_true_labels, train_pred_labels, average=None)
train_specificity = calculate_specificity(train_true_labels, train_pred_labels, num_classes)
train_auc = calculate_auc(train_true_labels, train_pred_proba, num_classes)

# validation set
val_precision = precision_score(val_true_labels, val_pred_labels, average=None)
val_recall = recall_score(val_true_labels, val_pred_labels, average=None)
val_f1 = f1_score(val_true_labels, val_pred_labels, average=None)
val_specificity = calculate_specificity(val_true_labels, val_pred_labels, num_classes)
val_auc = calculate_auc(val_true_labels, val_pred_proba, num_classes)

print("Training set metrics:")
for i in range(num_classes):
    print(f"Class {i}: Precision: {train_precision[i]}, Recall: {train_recall[i]}, F1: {train_f1[i]}, Specificity: {train_specificity[i]}, AUC: {train_auc[i]}")

print("\nValidation set metrics:")
for i in range(num_classes):
    print(f"Class {i}: Precision: {val_precision[i]}, Recall: {val_recall[i]}, F1: {val_f1[i]}, Specificity: {val_specificity[i]}, AUC: {val_auc[i]}")

#weighted_specificity
def calculate_weighted_specificity(y_true, y_pred, num_classes):
    specificity_scores = []
    class_counts = np.bincount(y_true, minlength=num_classes)  
    total_samples = np.sum(class_counts)  

    for i in range(num_classes):
        true_negatives = np.sum((y_true != i) & (y_pred != i))
        false_positives = np.sum((y_true != i) & (y_pred == i))
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        weighted_specificity = (specificity * class_counts[i]) / total_samples 
        specificity_scores.append(weighted_specificity)
    
    weighted_average_specificity = np.sum(specificity_scores)  
    return weighted_average_specificity

train_weighted_specificity = calculate_weighted_specificity(train_true_labels, train_pred_labels, num_classes)
val_weighted_specificity = calculate_weighted_specificity(val_true_labels, val_pred_labels, num_classes)

print(f"Training Weighted Average Specificity: {train_weighted_specificity}")
print(f"Validation Weighted Average Specificity: {val_weighted_specificity}")





