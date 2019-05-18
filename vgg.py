# using 60, 20, 20 split

import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from math import ceil
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import roc_curve,auc
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from IPython.display import display
from PIL import Image

target_size = 224, 224
batch_size = 16
class_labels_encoded = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
class_labels = ['safe_driving', 'texting_right', 'talking_on_phone_right', 'texting_left', 'talking_on_phone_left',
                'operating_radio', 'drinking', 'reaching_behind', 'doing_hair_makeup', 'talking_to_passanger']
num_classes = len(class_labels)

def create_top_model(activation_func, input_shape):
    
    model = Sequential()
    
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation=activation_func))
    
    return model

NUM_CLASSES = 10
data_path = './'

print("Starting splitting data.")
for i in range(NUM_CLASSES):
    
    curr_dir_path = data_path + 'c' + str(i) + '/'
    
    xtrain = labels = os.listdir(curr_dir_path)
    
    x, x_test, y, y_test = train_test_split(xtrain,labels,test_size=0.2,train_size=0.8)
    x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.25,train_size =0.75)
    
    for x in x_train:
        
        if (not os.path.exists('train/' + 'c' + str(i) + '/')):
            os.makedirs('train/' + 'c' + str(i) + '/')
            
        os.rename(data_path + 'c' + str(i) + '/' + x, 'train/' + 'c' + str(i) + '/' + x)
        
    for x in x_test:
        
        if (not os.path.exists('test/' + 'c' + str(i) + '/')):
            os.makedirs('test/' + 'c' + str(i) + '/')
            
        os.rename(data_path + 'c' + str(i) + '/' + x, 'test/' + 'c' + str(i) + '/' + x)
    
    for x in x_val:
        
        if (not os.path.exists('validation/' + 'c' + str(i) + '/')):
            os.makedirs('validation/' + 'c' + str(i) + '/')
            
        os.rename(data_path + 'c' + str(i) + '/' + x, 'validation/' + 'c' + str(i) + '/' + x)

print("Splitting data done.")
print("Starting feature extraction")

datagen = ImageDataGenerator(rescale=1./255)

# load vgg16 model, excluding the top fully connected layers
model = applications.VGG16(include_top=False, weights='imagenet')

# ---------- TRAINING DATA----------

# run training images through vgg and obtain its deep features (until last convolutional layer)
train_generator = datagen.flow_from_directory(
                    './train/',
                    target_size=target_size,
                    batch_size=batch_size,
                    class_mode=None, # data without labels
                    shuffle=False)

num_train_samples = len(train_generator.filenames)

# obtain steps required per epoch
train_steps = ceil(num_train_samples/batch_size)

# obtain deep/bottleneck features from vgg for the training data and save them
vgg_train_features = model.predict_generator(train_generator, steps=train_steps, verbose=1)
print('Saving deep features for training data...')
np.save('res/vgg_train_features.npy', vgg_train_features)

# ---------- VALIDATION DATA----------

# run validation images through vgg and obtain its deep features (until last convolutional layer)
val_generator = datagen.flow_from_directory(
                    './validation/',
                    target_size=target_size,
                    batch_size=batch_size,
                    class_mode=None, # data without labels
                    shuffle=False)

num_val_samples = len(val_generator.filenames)

# obtain steps required per epoch
val_steps = ceil(num_val_samples/batch_size)

# obtain deep/bottleneck features from vgg for the validation data and save them
vgg_val_features = model.predict_generator(val_generator, steps=val_steps, verbose=1)
print('Saving deep features for validation data...')
np.save('res/vgg_val_features.npy', vgg_val_features)

# ---------- TESTING DATA----------

# run testing images through vgg and obtain its deep features (until last convolutional layer)
test_generator = datagen.flow_from_directory(
                    './test/',
                    target_size=target_size,
                    batch_size=batch_size,
                    class_mode=None, # data without labels
                    shuffle=False)

num_test_samples = len(test_generator.filenames)

# obtain steps required per epoch
test_steps = ceil(num_test_samples/batch_size)

# obtain deep/bottleneck features from vgg for the testing data and save them
vgg_test_features = model.predict_generator(test_generator, steps=test_steps, verbose=1)
print('Saving deep features for testing data...')
np.save('res/vgg_test_features.npy', vgg_test_features)

print("Feature extraction done")

print("Training model now.")

# global variables
epochs = 50
datagen = ImageDataGenerator(rescale=1./225)

# ---------- LOAD TRAINING DATA ----------

# create datagen and train generator to load the data from directory
train_generator = datagen.flow_from_directory(
                            './train/',
                            target_size=target_size,
                            batch_size=batch_size,
                            class_mode='categorical',
                            shuffle=False) # data is ordered
                            
num_train_samples = len(train_generator.filenames)

# load vgg features
train_data = np.load('res/vgg_train_features.npy')

train_labels = train_generator.classes
train_labels_onehot = to_categorical(train_labels, num_classes=num_classes)

# ---------- LOAD VALIDATION DATA ----------

# create datagen and train generator to load the data from directory
val_generator = datagen.flow_from_directory(
                            './validation/',
                            target_size=target_size,
                            batch_size=batch_size,
                            class_mode='categorical',
                            shuffle=False) # data is ordered
                            
num_val_samples = len(val_generator.filenames)

# load vgg features
val_data = np.load('res/vgg_val_features.npy')

val_labels = val_generator.classes
val_labels_onehot = to_categorical(val_labels, num_classes=num_classes)

# ---------- CREATE AND TRAIN MODEL ----------

# create the top model to be trained
model = create_top_model("softmax", train_data.shape[1:])
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# only save the best weights. if the accuracy doesnt improve in 2 epochs, stop.
checkpoint_callback = ModelCheckpoint(
                        "res/top_model_weights.h5", # store weights with this file name
                        monitor='val_acc',
                        verbose=1,
                        save_best_only=True,
                        mode='max')

early_stop_callback = EarlyStopping(
                        monitor='val_acc',
                        patience=2, # max number of epochs to wait
                        mode='max') 

callbacks_list = [checkpoint_callback, early_stop_callback]

# train the model
history = model.fit(
            train_data,
            train_labels_onehot,
            epochs=epochs,
            batch_size=batch_size,
            # validation_data=val_data,
            validation_data=(val_data, val_labels_onehot),
            callbacks=callbacks_list)

print("Model training done.")
print("Testing image now.")


# ---------- FUNCTION DEFINITIONS ----------

def plot_confusion_matrix(y_true, y_pred):
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    title = 'Confusion matrix'
    
    # normalize matrix
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    plt.figure()
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels_encoded))
    plt.xticks(tick_marks, class_labels_encoded, rotation=0)
    plt.yticks(tick_marks, class_labels_encoded)

    threshold = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], '.2f'),
                 horizontalalignment='center',
                 color='white' if conf_matrix[i, j] > threshold else 'black')

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_roc(y_true, y_pred):
    
    # calculate roc and auc
    false_pos_rate = dict()
    true_pos_rate = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        false_pos_rate[i], true_pos_rate[i], _ = roc_curve(y_true[:,i], y_pred[:,i])
        roc_auc[i] = auc(false_pos_rate[i], true_pos_rate[i])
        
    # plot all
    
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, num_classes))
    
    for i, color in zip(range(num_classes), colors):
        plt.plot(false_pos_rate[i], true_pos_rate[i], lw=2, c=color,
                    label='c{0} (auc = {1:0.2f})'.format(i, roc_auc[i]))

    # plot random guess roc
    plt.plot([0, 1], [0, 1], 'k--',color='salmon', lw=2, label='Random Guess')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid()
    
    plt.show()

# ---------- GET TESTING DATA ----------

# create datagen and train generator to load the data from directory
datagen = ImageDataGenerator(rescale=1.0/255.0) 
test_generator = datagen.flow_from_directory(
                        './test/', 
                        target_size=target_size,
                        batch_size=batch_size,
                        class_mode='categorical', # specify categorical
                        shuffle=False) # data is ordered

# load vgg features
test_data = np.load('res/vgg_test_features.npy')

test_labels = test_generator.classes # actual class number
test_labels_onehot = to_categorical(test_labels, num_classes=num_classes) # class number in onehot

# ---------- TEST MODEL ----------

model = create_top_model("softmax", test_data.shape[1:])
model.load_weights("res/_top_model_weights.h5")  

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

predicted = model.predict_classes(test_data)

# ---------- DISPLAY INFORMATION----------

loss, acc = model.evaluate(test_data, test_labels_onehot, batch_size=batch_size, verbose=1)
print("loss: ", loss)
print("accuracy: {:8f}%".format(acc*100))
plot_confusion_matrix(test_labels, predicted)
predicted_onehot = to_categorical(predicted, num_classes=num_classes)
plot_roc(test_labels_onehot, predicted_onehot)