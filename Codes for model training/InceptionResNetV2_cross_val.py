# -*- coding: utf-8 -*-
"""
InceptionResNetV2 cross-validation
@author: Huibo Zhang
"""

import numpy as np 
import glob
import cv2
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

"""
###input data
"""
##### Training set:
print(os.listdir("Train_new2/train/"))

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
SIZE = 224

train_images = []
train_labels = [] 

for directory_path in glob.glob("Train_new2/train/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)
     
X_train = np.array(train_images)
X_train = X_train / 255.0
y_train = np.array(train_labels)
len(train_images)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

le.fit(train_labels) 
Y_train = le.transform(train_labels)


num_folds = 10
kfold = KFold(n_splits=num_folds, shuffle=True)
# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(X_train, Y_train):
    InceptionResNetV2_model = Sequential()
    pretrained_model= tf.keras.applications.InceptionResNetV2(include_top=False,
                   input_shape=(224,224,3),
                   pooling='avg',classes=3,
                   weights='imagenet')
    for layer in pretrained_model.layers:
            layer.trainable=False

    InceptionResNetV2_model.add(pretrained_model)
    InceptionResNetV2_model.add(Flatten())
    InceptionResNetV2_model.add(Dense(512, activation='relu'))
    InceptionResNetV2_model.add(Dense(3, activation='softmax'))
    InceptionResNetV2_model.summary()
    
    InceptionResNetV2_model.compile(optimizer='adam', loss = "sparse_categorical_crossentropy",metrics = ['accuracy'])
    checkpointer = tf.keras.callbacks.ModelCheckpoint("new_fold_" + str(fold_no) + "_" + 'InceptionResNetV2_model.h5', verbose=1, save_best_only=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs'),
        checkpointer]
    
    history = InceptionResNetV2_model.fit(X_train[train],Y_train[train],
                    epochs=50, 
                    validation_data=(X_train[test],Y_train[test]),
                    verbose = 1,
                    callbacks=callbacks)
    
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    # Generate generalization metrics
    scores = InceptionResNetV2_model.evaluate(X_train[test], Y_train[test], verbose=1)
    print(f'Score for fold {fold_no}: {InceptionResNetV2_model.metrics_names[0]} of {scores[0]}; {InceptionResNetV2_model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    # Increase fold number
    fold_no = fold_no + 1
    
# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


"""
######### prediction
"""

InceptionResNetV2_model=keras.models.load_model('best_InceptionResNetV2_model.h5')
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score, roc_auc_score#, confusion_matrix
#import numpy as np
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
        'Train_new3/train/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse',
        shuffle=False)

validation_generator = validation_datagen.flow_from_directory(
        'Train_new3/test/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse',
        shuffle=False)

train_steps = np.ceil(train_generator.samples / train_generator.batch_size)
val_steps = np.ceil(validation_generator.samples / validation_generator.batch_size)

train_pred_proba = predict_model(InceptionResNetV2_model, train_generator, steps=train_steps)
val_pred_proba = predict_model(InceptionResNetV2_model, validation_generator, steps=val_steps)

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

#training set
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

# weighted_specificity
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

from sklearn.metrics import confusion_matrix
# Calculate the confusion matrices for the training and validation sets
train_conf_matrix = confusion_matrix(train_true_labels, train_pred_labels)
val_conf_matrix = confusion_matrix(val_true_labels, val_pred_labels)
# Print the confusion matrices
print("Training Set Confusion Matrix:\n", train_conf_matrix)
print("Validation Set Confusion Matrix:\n", val_conf_matrix)

