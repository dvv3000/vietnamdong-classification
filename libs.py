import numpy as np
import cv2
import time
import os
import matplotlib.pyplot as plt
import random
import pickle
import sys
import seaborn as sns


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix



from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.models import  load_model

import tensorflow
from tensorflow.keras.layers import Input, BatchNormalization, ReLU, Conv2D, Dense, MaxPool2D, AvgPool2D, GlobalAvgPool2D, Concatenate, Dropout, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications import DenseNet121


