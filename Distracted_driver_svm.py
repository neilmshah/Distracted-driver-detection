#!/usr/bin/env python
# coding: utf-8

# In[24]:


from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
import skimage 
from skimage import io 
from skimage.transform import resize
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import os
#imported mahotas library for haralick feature descriptor
import mahotas
#imported opencv cv2 for color histogram and humoments
import cv2
#from sklearn import svm
#from sklearn import metrics
from sklearn.metrics import accuracy_score


# In[25]:


# declaring a class for collecting the data information and data path is set 
class Configuration:
    def __init__(self):
        self.maxwidth =0
        self.maxheight=0
        self.minwidth = 35000
        self.minheight = 35000
        self.imgcount=0
        self.img_width_adjust = 480
        self.img_height_adjust= 360
        self.data_dir = "/Users/Harshitha/Desktop/MachineLearning/Project/state-farm-distracted-driver-detection/imgs/train"


# In[26]:


#creating an config object 
config = Configuration()


# In[27]:


#function for finding the max,min  heights and max,min widths. It traverses throught the all folders and count
#number of images in the entire data. Update the respective values in the config object accordingly. 
def calculateDimension(path):
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg"):
                config.imgcount+=1
                filename = os.path.join(subdir, file)
                image = io.imread(filename)
                width = image.shape[0]
                height = image.shape[1]
                if width < config.minwidth:
                    config.minwidth = width
                if height < config.minheight:
                    config.minheight = height
                if width > config.maxwidth:
                    config.maxwidth = width
                if height > config.maxheight:
                    config.maxheight = height
    return


# In[28]:


calculateDimension(config.data_dir)
print("Minimum Width:\t",config.minwidth)
print("Minimum Height:\t",config.minheight)
print("Maximum Width:\t",config.maxwidth)
print("Maximum Height:\t",config.maxheight)
print("Image Count:\t",config.imgcount)


# In[29]:


fixed_size = tuple((64, 64))


# In[30]:


#used global feature descriptors HuMoments functions for gathering the shape details
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # extract the shape feature vector
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# In[31]:


#used global feature descriptors haralick functions for gatahering the texture details
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # extract texture feature vector. This function expects the image to be in gray scale
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick


# In[32]:


#used global feature descriptors colorhistogram functions for gathering the color details
bins = 8
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #extract color feature vector
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# In[33]:


def load_image_files(container_path,dimension=(64, 64)):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]
    
    print(categories)
    print(folders)
    print(image_dir)
    
    images = []
    flat_data = []
    target = []
    global_features = []
    for i, direc in enumerate(folders):
        # get the current training label
        current_label = direc.name
        for file in direc.iterdir():
            file = str(file)
            image = cv2.imread(file)
            #resizing the image to 64*64
            image = cv2.resize(image, dimension)
            target.append(current_label)
            # Global feature extraction is done here 
            fv_hu_moments = fd_hu_moments(image)
            fv_haralick   = fd_haralick(image)
            fv_histogram  = fd_histogram(image)
            #combing the global feautures  
            global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
            global_features.append(global_feature)
        print(current_label)
        print("feature extraction done with folder: {}".format(current_label))

    print("Entire Feature Extraction done")
    return global_features,target


# In[34]:


global_features,labels = load_image_files("/Users/Harshitha/Desktop/MachineLearning/Project/state-farm-distracted-driver-detection/imgs/train")


# In[35]:


#encoding the target labels 
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print("training labels encoded")
print(target)


# In[36]:


print(global_features[0].shape)


# In[37]:


# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("feature vector normalized")

print("target labels: {}".format(target))
print("target labels shape: {}".format(target.shape))

# get the overall feature vector size
print("feature vector size {}".format(np.array(global_features).shape))


# In[38]:


global_features = np.array(rescaled_features)
global_labels = np.array(target)


# In[39]:


(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(global_features,global_labels,
                                                                test_size=0.30,random_state=9)


# In[40]:


print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
print("Train labels: {}".format(trainLabelsGlobal.shape))
print("Test labels : {}".format(testLabelsGlobal.shape))


# In[41]:


clf = svm.SVC(C=5, gamma=10,kernel='rbf')
clf.fit(trainDataGlobal, trainLabelsGlobal)


# In[42]:


y_pred = clf.predict(testDataGlobal)
print(metrics.classification_report(testLabelsGlobal, y_pred))
print("Accuracy Score  : {}".format(accuracy_score(testLabelsGlobal, y_pred)))


# In[ ]:




