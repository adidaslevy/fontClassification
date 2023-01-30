import h5py
import pickle
import numpy as np
import PIL
import matplotlib.pyplot as plt
import cv2
import scipy
import itertools
import random
import keras
import os
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
BASE_PATH = "/home/alevy/studies/computerVision/Project"

BATCH_SIZE = 128
EPOCHS = 10

# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

class ImageProperties:
    def __init__(self, name, image, fonts, text, wordsBB, charsBB):
        self.name = name
        self.wordImages = []
        self.charImages = []
        self.wordFonts = []
        self.__extractFontsForWords__(fonts, text)
        self.wordsBB = wordsBB
        self.charsBB = charsBB
        # extract words
        self.__extractCharImages__(image, self.charsBB)
        self.fonts = fonts

    def __extractFontsForWords__(self, fonts, text):
        j = 0
        for i in range (text.shape[0]):
            wordLength = len(text[i])
            self.wordFonts.append(list(dict.fromkeys(fonts[j:j + wordLength]))[0])
            j += wordLength

    def __extractCharImages__(self, image, charBB):
        for i in range(charBB.shape[-1]):
            bb = charBB[:,:,i]
            bb = np.c_[bb,bb[:,0]]
            charImage = image[round(np.min(bb[1,:])):round(np.max(bb[1,:])), round(np.min(bb[0,:])):round(np.max(bb[0,:]))]
            #plt.imshow(charImage)
            #plt.show()
            self.charImages.append(charImage)

def openDB(dbPath):
    db = h5py.File(dbPath, 'r')
    return db

'''
Convert image to pil image, and resize to 36 * 36
'''
def openImage(img):
    im =PIL.Image.fromarray(np.uint8(img * 255)).convert('L')
    outImage = im.resize((32, 32))
    return outImage

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

'''
Organize and augment data
'''
def organaizeAndAugment(data):
    counter=0
    labels = []
    retVal = []

    for image in data:
        for i in range(len(image.charImages)):
            label = image.fonts[i]
            label = conv_label(label)
            pilImage = []
            try:
                pilImage = openImage(image.charImages[i])
            except:
                continue
            # Adding original image
            origImg = img_to_array(pilImage)
            retVal.append(origImg)
            labels.append(label)
    
            augument=["sharpen", "affine"]
            for l in range(0,len(augument)):
                a=itertools.combinations(augument, l+1)
                for i in list(a): 
                    combinations=list(i)
                    print(len(combinations))
                    temp_img = pilImage
                    for j in combinations:
                        if j == 'sharpen':
                            # Adding Blur image
                            temp_img = sharpen_image(temp_img)
                            #imshow(blur_img)
                        elif j == 'affine':
                            # Adding affine rotation image
                            temp_img = affine_rotation(origImg)
                    #plt.imshow(temp_img)
                    #plt.show()
                temp_img = img_to_array(temp_img)
                #plt.imshow(temp_img)
                #plt.show()
                retVal.append(temp_img)
                labels.append(label)
    return retVal, labels
        #plt.show()

'''
Data Augmentation section
'''
def noise_image(pil_im):
    # Adding Noise to image
    img_array = np.asarray(pil_im)
    mean = 0.0   # some constant
    std = 5   # some constant (standard deviation)
    noisy_img = img_array + np.random.normal(mean, std, img_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255)
    return noisy_img_clipped

def convertToGray(pil_im):
    grayImage = cv2.cvtColor(pil_im, cv2.COLOR_BGR2GRAY)
    #grayImage = grayImage.resize(36, 36)
    return grayImage

def sharpen_image(pil_im):
    #Adding Blur to image
    enhancer = PIL.ImageEnhance.Sharpness(pil_im)
    image_sharp = enhancer.enhance(2) 
    return image_sharp

def affine_rotation(img):
    rows, columns, mat = img.shape

    point1 = np.float32([[10, 10], [30, 10], [10, 30]])
    point2 = np.float32([[20, 15], [40, 10], [20, 40]])

    A = cv2.getAffineTransform(point1, point2)

    output = cv2.warpAffine(img, A, (columns, rows))
    return output

def gradient_fill(image):
    laplacian = cv2.Laplacian(image,cv2.CV_64F)
    laplacian = cv2.resize(laplacian, (105, 105))
    return laplacian

'''
Model
'''
def create_model():
    model = Sequential()
    #conv_inputs = Input(shape=(32, 32, 1)) # 32 # 64 
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')) #128 # 64
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten(name='flatten'))

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation='softmax', name='class'))

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

def prepareTrainData(data, labels):
    data = np.asarray(data, dtype="float") / 255.0
    labels = np.array(labels)
    print("Success")

    (trainX, testX, trainY, testY) = train_test_split(data,
        labels, test_size=0.25, random_state=42)
    trainY = to_categorical(trainY, num_classes=5)
    testY = to_categorical(testY, num_classes=5)
    aug = ImageDataGenerator(rotation_range=20, 
                             width_shift_range=0.1,
                             height_shift_range=0.1, 
                             shear_range=0.1, 
                             zoom_range=0.08,
                             horizontal_flip=False,
                             fill_mode='nearest')
    return trainX, testX, trainY, testY, aug

def prepareModelParams(model):
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1), optimizer=sgd, metrics=['accuracy'])
    early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')

    filepath=os.path.join(BASE_PATH, "top_model.h5")

    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [early_stopping, checkpoint]
    return callbacks_list

def runModel(model, trainX, trainY, testX, testY, callbacks_list):
    model.fit(trainX, trainY,shuffle=True,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(testX, testY),callbacks=callbacks_list)
    score = model.evaluate(testX, testY, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

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

def main():
    rawDataPath = os.path.join(BASE_PATH, "SynthText_train.h5")
    
    # See if there is a pickle of the data, if not run the read and augmentation,
    if not checkAugmentedDataPickle(BASE_PATH):
        db = openDB(rawDataPath)
        data = readDB(db)
        augmentedData, lables = organaizeAndAugment(data)
        pickleData(augmentedData, AUGMENTED_DATA_PICKLE)
        pickleData(lables, LABELS_PICKLE)
    else:
        augmentedData = getPickle(AUGMENTED_DATA_PICKLE)
        labels = getPickle(LABELS_PICKLE)
        with sess.as_default():
            trainX, testX, trainY, testY, aug = prepareTrainData(augmentedData, labels)
            K.set_image_data_format('channels_last')
            model = create_model()
            model.summary()
            callback_list = prepareModelParams(model)
            runModel(model, trainX, trainY, testX, testY, callback_list)
        sess.close()
        print("yalla!")
        

if __name__ == "__main__":
    main()