from enum import Enum
import numpy as np

class TestData:
    def __init__(self, name, image, text, charsBB):
        self.name = name
        self.charImages = []
        self.charsBB = charsBB
        self.text = text
        self.__extractCharImages__(image, self.charsBB)
        
    def __extractCharImages__(self, image, charBB):
        for i in range(charBB.shape[-1]):
            bb = charBB[:,:,i]
            bb = np.c_[bb,bb[:,0]]
            charImage = image[round(np.min(bb[1,:])):round(np.max(bb[1,:])), round(np.min(bb[0,:])):round(np.max(bb[0,:]))]
            self.charImages.append(charImage)

class TrainData(TestData):
    def __init__(self, name, image, fonts, text, charsBB):
        super().__init__(name, image, text, charsBB)
        self.fonts = fonts

class WorkMode(Enum):
    Train = 0
    Test = 1

class DataSplit(Enum):
    Train = 0
    Validation = 1
