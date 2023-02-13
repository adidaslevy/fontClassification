import os
import pickle
'''
Pickle management
'''
LABELS_PICKLE = 'labels.pickle'

def checkLabelsPickle(path):
    return os.path.exists(os.path.join(path, LABELS_PICKLE))

def pickleData(data, path):
    pickle_out_data = open(os.path.join(path, LABELS_PICKLE), "wb")
    pickle.dump(np.array(data), pickle_out_data)
    pickle_out_data.close()

def getPickle(path):
    pickleIn = open(os.path.join(path, LABELS_PICKLE), "rb")
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

IMAGE_LENGTH = 32
IMAGE_WIDTH = 32

def isValidFile(parser, arg):
    arg = arg.lstrip()
    if not os.path.exists(arg):
        parser.error("The file {arg} does not exist.")
    else:
        return arg