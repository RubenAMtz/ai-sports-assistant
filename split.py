import pandas as pd
import numpy as np
import os, os.path
import math
from random import shuffle
import random
from shutil import copyfile

SEED = 42
class Split:

    def __init__(self, train_split, test_split):
        train, test, images, annotations = self.shuffle_and_indexes(train_split, test_split)
        self._train_indexes = train
        self._test_indexes = test
        self._images = images
        self._annotations = annotations
        self.copy_and_move(train, test, images, annotations)

    def shuffle_and_indexes(self, train_split, test_split):
        path, dirs, images = next(os.walk("./images/"))
        path2, dirs2, annotations = next(os.walk("./annotations/"))
        random.seed(SEED)
        shuffle(images)
        random.seed(SEED)
        shuffle(annotations)
        train_index = list(range(math.ceil(len(images)*train_split/100)))
        test_index = list(range(train_index[-1]+1,len(images)))
        return train_index, test_index, images, annotations

    def copy_and_move(self, train_indexes, test_indexes, images, annotations):
        images_train = [images[i] for i in train_indexes]
        images_test = [images[i] for i in test_indexes]
        print(images_train)
        print(images_test)
        annotations_train = [annotations[i] for i in train_indexes]
        annotations_test = [annotations[i] for i in test_indexes]
        for image in images_train:
            copyfile('./images/'+ str(image), './images/train/'+ str(image))
        for image in images_test:
            copyfile('./images/'+ str(image), './images/test/'+ str(image))
        for annotation in annotations_train:
            copyfile('./annotations/'+ str(annotation), './images/train/'+ str(annotation))
        for annotation in annotations_test:
            copyfile('./annotations/'+ str(annotation), './images/test/'+ str(annotation))

a = Split(80,20)
print(a)