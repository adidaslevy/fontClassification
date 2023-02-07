import argparse
from argparse import ArgumentParser
from cmath import exp
import pandas as pd
import h5py
import pickle
import numpy as np
import PIL
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import cv2
import scipy
import itertools
import random
import keras
import os
import sys
import uuid
from keras import optimizers
from keras.utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import callbacks
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D , UpSampling2D ,Conv2DTranspose
from keras import backend as K
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from tensorflow import keras
import os.path
AUGMENTED_DATA_PICKLE = "augmentedData.pickle"
LABELS_PICKLE = "labels.pickle"
WORKING_DIR = ""

BATCH_SIZE = 128
IMAGE_LENGTH = 32
IMAGE_WIDTH = 32
EPOCHS = 50

# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

class ImageProperties:
    def __init__(self, name, image, fonts, text, wordsBB, charsBB):
        self.name = name
        self.charImages = []
        self.predictedFonts = []
        self.charsBB = charsBB
        self.__extractCharImages__(image, self.charsBB)
        self.fonts = fonts

    def __extractCharImages__(self, image, charBB):
        for i in range(charBB.shape[-1]):
            bb = charBB[:,:,i]
            bb = np.c_[bb,bb[:,0]]
            charImage = image[round(np.min(bb[1,:])):round(np.max(bb[1,:])), round(np.min(bb[0,:])):round(np.max(bb[0,:]))]
            self.charImages.append(charImage)

def openDB(dbPath):
    db = h5py.File(dbPath, 'r')
    return db



'''
Return a tuple of arrays in the following form:
imageName, image, fonts for that image, words
'''
def readDB(db):
    data = []
    imageNames = sorted(db['data'].keys())
    for i in imageNames:
        name = i
        image = db['data'][i][...]
        charBB = db['data'][i].attrs['charBB']
        wordBB = db['data'][i].attrs['wordBB']
        fonts = db['data'][i].attrs['font']
        text = db['data'][i].attrs['txt']
        image = ImageProperties(name, 
                                image, 
                                fonts, 
                                text,
                                wordBB,
                                charBB)
        data.append(image)
    return data

'''
Get font list
'''
def conv_label(label):
    if label == b'Alex Brush':
        return 0
    elif label == b'Open Sans':
        return 1
    elif label == b'Sansation':
        return 2
    elif label == b'Ubuntu Mono':
        return 3
    elif label == b'Titillium Web':
        return 4


def openImage(img):
    im =PIL.Image.fromarray(img)
    sharpImage = sharpenImage(im)
    outImage = sharpImage.resize((IMAGE_LENGTH, IMAGE_WIDTH), resample=PIL.Image.BICUBIC).convert('L')
    return outImage


'''
Organize and augment data
'''
def organaizeData(data: ImageProperties, workingDir: str):
    counter=0
    if (os.path.exists(os.path.join(WORKING_DIR, 'dataset'))):
        return
    dataSet = {"images": [], "filenames": [], "labels": []}
    
    for image in data:
        for i in range(len(image.charImages)):
            label = image.fonts[i]
            #label = conv_label(label)
            pilImage = []
            try:
                pilImage = openImage(image.charImages[i])
            except:
                continue
            # Adding original image
            origImg = img_to_array(pilImage)
            dataSet['images'].append(pilImage)
            dataSet['labels'].append(label)
    
    # save char images in dataset dir
    datasetDir = os.path.join(workingDir, 'dataset')
    try:
        os.mkdir(datasetDir)
    except (FileExistsError):
        print("Dir: {} already exists".format(datasetDir))
    labelsSet = set(dataSet['labels'])
    for label in labelsSet:    
        labelDirPath = os.path.join(datasetDir, str(label)) 
        try:   
            os.mkdir(labelDirPath)
        except (FileExistsError):
            print("Dir: {} already exists".format(labelDirPath))
    for image, label in zip(dataSet['images'], dataSet['labels']):
        fileName = "img_" + str(uuid.uuid4()) + ".jpg"
        labelDir = os.path.join(datasetDir, str(label))
        imageFileName = os.path.join(labelDir, fileName)
        image.save(imageFileName)
        dataSet["filenames"].append(imageFileName)
    
    dataSet.pop('images')
    return pd.DataFrame(dataSet)

'''
Augmentation methods
'''
def sharpenImage(pil_im):
    #Sharpen Image
    enhancer = PIL.ImageEnhance.Sharpness(pil_im)
    image_sharp = enhancer.enhance(2) 
    return image_sharp

def augment(dataSet: dict, n: int, workingDir: str = WORKING_DIR, imgSize: tuple = ((IMAGE_LENGTH, IMAGE_WIDTH))) -> dict:
        datasetDir = os.path.join(workingDir, 'dataset')
        # create and store the augmented images  
        total=0
        dataGenerator=ImageDataGenerator(rotation_range=20, 
                             width_shift_range=0.2,
                             height_shift_range=0.2, 
                             shear_range=0.1, 
                             zoom_range=0.08,
                             horizontal_flip=False,
                             fill_mode='nearest')
         
        groups=dataSet.groupby('labels')
        for label in dataSet['labels'].unique():

            # for every group count how many are currently in class
            # if there are less than total desired count
            # generate more by dataGenerator
            group = groups.get_group(label) 
            sampleCount=len(group) 

            if (sampleCount < n): 
                augmentedImageCount = 0

                # Number of images to create
                delta = n - sampleCount
                targetDir = os.path.join(datasetDir, str(label))
                msg = '{0:40s} for class {1:^30s} creating {2:^5s} augmented images'.format(' ', str(label), str(delta))
                print(msg, '\r', end='') 

                aug_gen=dataGenerator.flow_from_dataframe(group, directory=WORKING_DIR, x_col='filenames', y_col="label", target_size=imgSize,
                                                class_mode=None, batch_size=1, shuffle=True, seed = 42, 
                                                save_to_dir=targetDir, save_prefix='aug-', color_mode='grayscale',
                                                save_format='jpg')
                while augmentedImageCount < delta:
                    images=next(aug_gen)            
                    augmentedImageCount += len(images)
                total +=augmentedImageCount
        print('Total Augmented images created= ', total)

def trainModel(inputFile):
    WORKING_DIR = os.path.dirname(inputFile)
    if (not os.path.exists(os.path.join(WORKING_DIR, 'dataset'))):
        db = openDB(inputFile)
        data = readDB(db)
        df = organaizeData(data, WORKING_DIR)
        augment(df, 8000, WORKING_DIR)

    history = []
    #with sess.as_default(WORKING_DIR):
    train_ds, val_ds = prepareTrainData(WORKING_DIR)
    K.set_image_data_format('channels_last')
    model = create_model()
    model.summary()
    callback_list = prepareModelParams(model, train_ds)
    history = runModel(model, train_ds, val_ds, callback_list)
    #sess.close()
    summarizeModel(history)
    print("yalla!")

'''
Model
'''
def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv2D(filter, (1,1), strides = (2,2))(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x

def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x

def ResNet34(shape = (32, 32, 1), classes = 10):
    # Step 1 (Setup Input Layer)
    x_input = tf.keras.layers.Input(shape)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # Step 4 End Dense Network
    x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    x = tf.keras.layers.Dense(5, activation = 'softmax')(x)
    model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet34")
    return model

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMAGE_LENGTH, IMAGE_WIDTH, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten(name='flatten'))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation='softmax'))


    #model = keras.Model(inputs=conv_inputs, outputs=conv_outputs)
    '''
    model=Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Dense(5, activation='softmax'))
    '''
    # Cu Layers 
    '''
    model.add(Conv2D(64, kernel_size=(48, 48), activation='relu', input_shape=(36,36,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(24, 24), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2DTranspose(128, (24,24), strides = (2,2), activation = 'relu', padding='same', kernel_initializer='uniform'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2DTranspose(64, (12,12), strides = (2,2), activation = 'relu', padding='same', kernel_initializer='uniform'))
    model.add(UpSampling2D(size=(2, 2)))

    #Cs Layers
    model.add(Conv2D(256, kernel_size=(12, 12), activation='relu'))
    model.add(Conv2D(256, kernel_size=(12, 12), activation='relu'))
    model.add(Conv2D(256, kernel_size=(12, 12), activation='relu'))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2383,activation='relu'))
    model.add(Dense(5, activation='softmax'))
    '''
    return model

def prepareTrainData(workingDir: str):
    # load data and from HD
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(workingDir, 'dataset'),
        validation_split=0.25,
        subset="training",
        color_mode='grayscale',
        label_mode='categorical',
        seed=42,
        shuffle=True,
        image_size=(IMAGE_LENGTH, IMAGE_WIDTH),
        batch_size=BATCH_SIZE)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(workingDir, 'dataset'),
        validation_split=0.25,
        subset="validation",
        color_mode='grayscale',
        label_mode='categorical',
        seed=42,
        shuffle=True,
        image_size=(IMAGE_LENGTH, IMAGE_WIDTH),
        batch_size=BATCH_SIZE)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds

def exp_decay(epoch):
   initial_lrate = 0.01
   k = 0.1
   lrate = initial_lrate * tf.math.exp(-k*epoch)
   return lrate
    
def prepareModelParams(model, train_ds):
    lr = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=400, decay_rate=0.98, staircase=True)

    sgd = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=False), 
                  optimizer=sgd, metrics=['accuracy'])
    early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='min')
    learningRateScheduler = callbacks.LearningRateScheduler(exp_decay)
    filepath=os.path.join(WORKING_DIR, "top_model.h5")

    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [early_stopping, checkpoint, learningRateScheduler]
    return callbacks_list

def runModel(model, train_ds, val_ds, callbacks_list):
    history = model.fit(train_ds, shuffle=True,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=val_ds,callbacks=callbacks_list)
    score = model.evaluate(val_ds, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return history

def summarizeModel(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

'''
Pickle management
'''

def checkAugmentedDataPickle(path):
    return os.path.exists(os.path.join(path, AUGMENTED_DATA_PICKLE)) and \
        os.path.exists(os.path.join(path, LABELS_PICKLE))

def pickleData(data, path):
    pickle_out_data = open(os.path.join(BASE_PATH, path), "wb")
    pickle.dump(data, pickle_out_data)
    pickle_out_data.close()

def getPickle(path):
    pickleIn = open(os.path.join(BASE_PATH, path), "rb")
    return pickle.load(pickleIn)

def findMaxLabel(labels):
    counter = -1
    maxLabel = -1
    for i in range (0, 5):
        print("label {} is found: {}".format(i, labels.count(i)))
        if (labels.count(i) > counter):
            maxLabel = i
            counter = labels.count(i)
    print("{} label is found max: {}".format(maxLabel, counter))
    return maxLabel, counter

def isValidFile(parser, arg):
    arg = arg.lstrip()
    if not os.path.exists(arg):
        parser.error("The file {arg} does not exist.")
    else:
        return arg

def main(argv):
    '''
    Argument parser
    '''
    parser = ArgumentParser(description="Train and test font classification model")
    group1 = parser.add_argument_group("group1")
    group1.add_argument("-t", action = 'store_true', help = "This option is used to train model using h5 file as an input")
    iFile = group1.add_argument("-i", type = lambda x: isValidFile(parser, x), dest="inputFileTrain",
                        help="Train model using train h5 DB filepath", metavar="FILE")
    
    group2 = parser.add_argument_group("group2")
    group2.add_argument("-p", action = 'store_true', help = "This option is used to predict model using h5 file as an input, and output a csv")
    group2._group_actions.append(iFile)
    group2.add_argument("-o", type = argparse.FileType('w'), help = "Output csv file")
                        
    args = parser.parse_args()

    '''
    Train Model
    '''
    if (args.t):
        trainModel(args.inputFileTrain)
    elif(args.p):
        trainModel(args.inputFileTrain)

if __name__ == "__main__":
    main(sys.argv[1:])