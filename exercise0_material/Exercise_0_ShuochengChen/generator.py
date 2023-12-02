import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        self.epoch_counter = 0
        self.batch_cursor = 0
        self.data = []

        #self.end_marker = False

        with open(os.path.join('./data', label_path), 'r') as f:
            self.labels = json.load(f)

        for key in self.labels:
            #self.data = (np.array(scipy.misc.imread(os.path.join(file_path, key))).astype(np.float32) / 255.0).reshape(1, 32, 32, 3)
            self.data.append((np.array(np.load(os.path.join('./data', file_path, key+'.npy'))), self.labels[key]))
            # data = (np.ndarray, int_label)

        if self.shuffle:
            import random
            random.shuffle(self.data)
        self.data_size = len(self.labels)


    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method

        # bs = 11
        # 99  (0:11, 11:22, 22:33, 33:44, ..., 88:99, 99:110)
        # 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        # cursor + bs = 110, 110 - ds = 10, 10 - 1 = 9
        temp_bs = self.batch_size
        batch = []
        while (self.batch_cursor + temp_bs > self.data_size):
            #if self.end_marker: 
            #    self.end_marker = False
            #    self.epoch_counter = self.epoch_counter + 1
            #    self.batch_cursor = 0
            temp_bs = temp_bs - (self.data_size - self.batch_cursor)
            batch += self.data[self.batch_cursor: self.data_size]

            self.epoch_counter = self.epoch_counter + 1
            self.batch_cursor = 0
            # self.end_marker = True

            if self.shuffle:
                import random
                random.shuffle(self.data)
            # bc = 90 bs = 12 ds = 100
            # t_bs = 2, bc = 0, append(90:100)

        if temp_bs != 0:
            batch += self.data[self.batch_cursor: self.batch_cursor + temp_bs]
            self.batch_cursor = self.batch_cursor + temp_bs
            temp_bs = 0

        images, labels = map(list, zip(*batch))
#        if self.batch_cursor + self.batch_size > self.data_size:
#            batch = self.data[self.batch_cursor: self.batch_size] + self.data[0: self.batch_cursor + self.batch_size - self.data_size - 1]
#            self.batch_cursor = 0

        import skimage.transform as sktransform
        for i in range(len(images)):
            images[i] = sktransform.resize(images[i], self.image_size)
            images[i] = self.augment(images[i])

        #TODO: 
        return np.asarray(images), np.asarray(labels)

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        import random
        if self.mirroring:
            if random.randint(0, 1):
                img = np.flip(img, 0)
            if random.randint(0, 1):
                img = np.flip(img, 1)
        if self.rotation:
            if random.randint(0, 1):
                img = np.rot90(img, random.randint(1, 3))
        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch_counter

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict[x]
    
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        images, labels = self.next()
        label_names = []
        for key in labels:
            label_names.append(self.class_name(key))
        fig = plt.figure(figsize = (self.image_size[0], self.image_size[0]))
        for i in range(len(images)):
            ax = fig.add_subplot(len(images), 1, i+1)
            ax.set_title(label_names[i])
            ax.imshow(images[i])
            ax.axis('off')

        plt.show()

