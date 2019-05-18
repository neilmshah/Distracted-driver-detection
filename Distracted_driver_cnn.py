import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

class Distracted_Driver():
    def __init__(self):
        self.epochs = 10
        self.batch_size = 16
        self.maxwidth =0
        self.maxheight=0
        self.minwidth = 35000
        self.minheight = 35000
        self.imgcount=0
        self.img_width_adjust = 480
        self.img_height_adjust= 360
        # Data downloaded from kaggle to this folder
        self.data_dir = "train/"
        # load a sample image from the train data set
        self.load_image()
        self.dirCount=self.listDirectoryCounts()
        self.categoryInfo = self.printCategoryInfo()
        self.sortByCategory()
        self.findMaxAndMinDimensions()
        # split data into train and validation sets
        self.train_generator, self.val_generator = self.setup_data()
        self.model = self.build_model()
        # fit the model  in batches  
        # number of epoch times
        self.fit_model()
        # Evaluate your model.
        self.eval_model()

       

    def load_image(self):
        print("loading a sample image from train data set")
        img=mpimg.imread(self.data_dir+'c0/img_4013.jpg')
        imgplot = plt.imshow(img)
        img.shape
        plt.show()
    
    " count all the sub folders inside train folder"
    " count the no. of files of each distracted driver catgeory"
    def listDirectoryCounts(self):
        dirlist = []
        for subdir, dirs, files in os.walk(self.data_dir,topdown=False):
            filecount = len(files)
            dirname = subdir
            dirlist.append((dirname,filecount))
        return dirlist

    "print the no. of images in each of the categories c0-c9"
    def printCategoryInfo(self):
        categoryInfo = pd.DataFrame(self.dirCount, columns=['Category','Count'])
        print(categoryInfo)
        print(categoryInfo.head)
        return categoryInfo

    def SplitCat(self):
        print("splitting into different categories")
        for index, row in self.categoryInfo.iterrows():
            directory=row['Category'].split('/')
            if directory[2]!='':
                directory=directory[2]
                self.categoryInfo.at[index,'Category']=directory
            else:
                self.categoryInfo.drop(index, inplace=True)
        return


    def sortByCategory(self):
        print("sorting files from c0 - c9")
        self.SplitCat()
        cinfo=self.categoryInfo.sort_values(by=['Category'])
        print(cinfo.to_string(index=False))
        self.categoryInfo = cinfo


    def findPictureDims(self):
        for subdir, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".jpg"):
                    self.imgcount+=1
                    filename = os.path.join(subdir, file)
                    image = Image.open(filename)
                    width, height = image.size
                    if width < self.minwidth:
                        self.minwidth = width
                    if height < self.minheight:
                        self.minheight = height
                    if width > self.maxwidth:
                        self.maxwidth = width
                    if height > self.maxheight:
                        self.maxheight = height 
        return

    "find maximum and minimum height, width to resize all the images"
    def findMaxAndMinDimensions(self):
        self.findPictureDims()
        print("Image Count:\t",self.imgcount)
        print("Minimum Width:\t",self.minwidth, "\tMinimum Height:",self.minheight)
        print("Maximum Width:\t",self.maxwidth, "\tMaximum Height:",self.maxheight)


    "building the model with 4 conv layers (conv2D+ maxpool2D) and 1 fully connected layer"
    "ReLU activation for all the layers"
    def build_model(self):
    
        inputs = Input(shape=(self.img_width_adjust,self.img_height_adjust,3), name="input")
        
        #Convolution 1 , 128 filiters
        conv1 = Conv2D(128, kernel_size=(3,3), activation="relu", name="conv_1")(inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2), name="pool_1")(conv1)

        #Convolution 2, 64 filters
        conv2 = Conv2D(64, kernel_size=(3,3), activation="relu", name="conv_2")(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), name="pool_2")(conv2)
        
        #Convolution 3, 32 filters
        conv3 = Conv2D(32, kernel_size=(3,3), activation="relu", name="conv_3")(pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2), name="pool_3")(conv3)
        
        #Convolution 4, 16 filters
        conv4 = Conv2D(16, kernel_size=(3,3), activation="relu", name="conv_4")(pool3)
        pool4 = MaxPooling2D(pool_size=(2, 2), name="pool_4")(conv4)
        
        #Fully Connected Layer
        flatten = Flatten()(pool4)
        fc1 = Dense(1024, activation="relu", name="fc_1")(flatten)
        
        #use softmax activation for output layer
        output=Dense(10, activation="softmax", name ="softmax")(fc1)
        
        # finalize and compile
        model = Model(inputs=inputs, outputs=output)
        # calculate loss as categorical cross entropy , as there are 10 categories
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
        print (model.summary())
        return model

    " split into train - 80% and validation - 20 %"
    def setup_data(self):
        train_datagen = ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)
        
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_width_adjust, self.img_height_adjust),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training')
        
        validation_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_width_adjust, self.img_height_adjust),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation')
        return train_generator, validation_generator

    "fit the model in batches of size 16"
    def fit_model(self):
        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.batch_size,
            epochs=self.epochs,
            validation_data=self.val_generator,
            validation_steps=self.val_generator.samples // self.batch_size,
            verbose=1)
        

    "After training the model with 10 epochs, evaluate the model, against validation data"
    def eval_model(self):
        scores = self.model.evaluate_generator(self.val_generator, steps=self.val_generator.samples // self.batch_size)
        print("Loss: " + str(scores[0]) + " Accuracy: " + str(scores[1]))


if __name__ == "__main__":
    d = Distracted_Driver()
    

    