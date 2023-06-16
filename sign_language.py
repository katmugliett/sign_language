import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import tensorflow as tf
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


Xdata=np.load('X.npy')
Ydata=np.load('Y.npy')

idxes=[0, 50, 100, 200, 300, 
       400, 500, 550, 600, 700, 
       800, 900, 1000, 1100, 1150, 
       1200, 1300, 1400, 1500, 1600, 
       1700, 1800, 1900, 2000, 2061]

fig, axes = plt.subplots(5, 5, figsize=(64, 64))

images=[]
for i in range(25):
    image=Xdata[idxes[i]]
    images.append(image)

for ax, img in zip(axes.flat, images):
    ax.imshow(img, interpolation='nearest', cmap='bone')
    ax.axis('off')  # Turn off axis labels and ticks

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()

X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata, test_size=0.33, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

g = sns.countplot(y_train)
plt.show()

print(Xdata.shape)
print(X_train[0].shape)
print(X_train[0])

image_sizes=[]
for i, sub in enumerate(Xdata):
    width, height = Xdata[i].shape
    image_sizes.append((width, height))

widths=[size[0] for size in image_sizes]
heights=[size[1] for size in image_sizes]

plt.hist(widths)
plt.show()
plt.hist(heights)
plt.show()

#scale pixel values to a smaller range to improve convergence
scaledAll = []
def pxl_scale(imgData):
    for i, sub in enumerate(imgData):
        minVal=np.min(imgData[i])
        maxVal=np.max(imgData[i])
        scaledImg = (imgData[i]-minVal)/(maxVal-minVal)
        scaledAll.append(scaledImg)
    return scaledAll

scaledAll=pxl_scale(Xdata)

fig, axes = plt.subplots(5, 5, figsize=(64, 64))

images=[]
for i in range(25):
    image=scaledAll[idxes[i]]
    images.append(image)

for ax, img in zip(axes.flat, images):
    ax.imshow(img, interpolation='nearest', cmap='bone')
    ax.axis('off')  # Turn off axis labels and ticks

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()


def pxlIntensPlot(ogData, normData):
    ogPixel=ogData[0].flatten() #flatten images into 1D array
    normPixel=normData[0].flatten()
    return ogPixel, normPixel

ogPixel, normPixel=pxlIntensPlot(Xdata, scaledAll)

plt.subplot(1, 2, 1)
plt.hist(ogPixel)
plt.subplot(1, 2, 2)
plt.hist(normPixel)

plt.tight_layout()
plt.show()

maxValOG=np.max(Xdata[0])
maxValNorm=np.max(scaledAll[0])

print(maxValOG)
print(maxValNorm)

#from pixel intensity plot it can be seen that pixels were already within a normalized range, so further scaling may be redundant.

#Reshape data with height=64, width=64, and 3rd dimension = 1: 28x28x1
#Keras required 3rd dimension to correspond to channels. 

#X_train_df=pd.DataFrame(X_train)
#X_test_df=pd.DataFrame(X_test)

##TO BE FIXED:
#for i, sub in enumerate(X_train):
#    X_train[i]=X_train[i].reshape(-1, 64, 64, 1)
#for i, sub in enumerate(X_test):
#    X_test[i]=X_test[i].reshape(-1, 28, 28, 1)
#print(X_train[0].shape)
#print(X_test_df[0].shape)

#One-hot encoding
y_train=to_categorical(y_train, num_classes=10)