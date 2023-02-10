import os
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