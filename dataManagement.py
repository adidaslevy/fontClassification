from enum import Enum
from matplotlib import pyplot as plt
from utils import isValidFile, IMAGE_LENGTH, IMAGE_WIDTH
import numpy as np
from PIL import Image, ImageEnhance
from shapely.geometry import Polygon

class TestData:
    def __init__(self, name, image, text, charsBB):
        self.name = name
        self.charImages = []
        self.charsBB = charsBB
        self.text = text
        #patches,gt_str = self.__extract__(image, self.charsBB,self.text)
        self.__extractCharImages__(image, self.charsBB)

    
    def __extractCharImages__(self, image, charBB):
        im = Image.fromarray(np.uint8(1-image*255))
        for i in range(charBB.shape[-1]):
            points = [(charBB[0][point][i], charBB[1][point][i]) for point in range(charBB.shape[1])]
            polygon = Polygon(points)
            charImage = im.crop(box = polygon.bounds)
            sharpImage = sharpenImage(charImage)
            outImage = sharpImage.resize((IMAGE_LENGTH, IMAGE_WIDTH), resample=Image.BICUBIC).convert('L')
            self.charImages.append(outImage)

def sharpenImage(pil_im):
    #Sharpen Image
    enhancer = ImageEnhance.Sharpness(pil_im)
    image_sharp = enhancer.enhance(2) 
    return image_sharp

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
