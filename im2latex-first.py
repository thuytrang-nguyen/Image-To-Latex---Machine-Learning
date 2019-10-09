# CS 352 - Machine Learning
# Final Project
# Due: December 23rd, 2018
# Malcolm Gilbert and Weronika Nguyen

# This is a model for training and predicting a the first character of a LateX formula

import sys
import numpy as np
from skimage import io
from scipy.misc import imresize
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import re
from utils import tokenize_formula, remove_invisible, normalize_formula

tokens_size = 0
np.set_printoptions(threshold=np.inf)

class img2latexFirst:

    def __init__(self, targs, imgs, test_split=0.1, learning_rate=0.001):
        '''Initialize classes and image size.'''
        self.learning_rate = learning_rate
        self.image_data = []
        self.targets = targs
        self.images = imgs
        self.test_split = test_split

    def _process_image(self, image):

        '''Processing images - first load, crop and resize image. Add to the self.image_data array.'''
        # load in image
        old_img = io.imread('./train-data/'+image+'.png')
        img1 = old_img[:,:,3]
        img = img1.reshape([50, 60, 1])
        # add image to image_data
        self.image_data.append(np.array(img))

    def prepare_data(self):
        '''processes all the input images by calling _process_image'''
        for image in self.images:
            self._process_image(image)

    def process_test_img(self, image):
        '''processes a single testing image and returns it'''
        old_img = io.imread('./data/'+image+'.png')
        img = old_img[:,:,3]
        img = img.reshape([-1, 50, 60, 1])
        return img

    def build_model(self):
        convnet = input_data(shape=[None, 50, 60, 1], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu', regularizer="L2")
        convnet = max_pool_2d(convnet, 2)
        convnet = local_response_normalization(convnet)
        convnet = conv_2d(convnet, 64, 3, activation='relu', regularizer="L2")
        convnet = max_pool_2d(convnet, 2)
        convnet = local_response_normalization(convnet)
        convnet = fully_connected(convnet, 128, activation='tanh')
        convnet = dropout(convnet, 0.8)
        convnet = fully_connected(convnet, 256, activation='tanh')
        convnet = dropout(convnet, 0.8)
        convnet = fully_connected(convnet, 4, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=0.01,
           loss='categorical_crossentropy', name='target')

        model = tflearn.DNN(convnet,
                            tensorboard_dir='log',
                            tensorboard_verbose=3
        )
        return model
        
    def train_model(self, model_name, epochs=5, batch_size=32):

        x = self.image_data
        y = self.targets
        X_train = np.asarray(x)
        y_train = np.asarray(y)
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        model = self.build_model()
        
        model.fit(X_train, y_train,
                  n_epoch=epochs,
                  shuffle=True,
                  validation_set=0.1,
                  show_metric=True,
                  batch_size=batch_size)
        
        self.model = model
        #self.model.save(model_name+".tfl")

    def load_model(self, model_file):
        model = self.build_model()
        model.load(model_file+".tfl")
        self.model = model

    def predict_formula(self, image, formula):
        # process testing img
        img = self.process_test_img(image)
        # get the prediction
        old_result = self.model.predict(img)
        result = np.reshape(old_result, 4)
        maxed = np.zeros(4)
        max_val = max(result)
        index_of_max = np.nonzero(result == max_val)[0]
        maxed[index_of_max] = 1

        correct = "yes"
        for i in range(len(maxed)):
          if (maxed[i] != formula[i]):
            correct = "no"
        return maxed, formula, correct

if __name__ == '__main__':

    first = {0:"\\int", 1:"\\sum", 2:"\\frac", 3:"\\sqrt"}

    f =  open('train-data.txt', "r", errors='ignore')
    formulas = f.readlines()
    f.close()
    
    # process training data
    training_targets = []
    training_images = []

    # loop through each formula, remove invisible characters, normalize and turn into tokens
    # get the first token and turn it into one-hot vector of size 4 
    for j in range(len(formulas)):
      i = remove_invisible(formulas[j])
      n = normalize_formula(i)
      t = tokenize_formula(n)
      row = [0,0,0,0];
      for j in range(len(first)):
        if (t[0]==first.get(j)):
          row[j] = 1;
      training_targets.append(row);

    for i in range(len(training_targets)):
        training_images.append(str(i))
    
    # load testing data
    f =  open('equations.txt', "r", errors='ignore')
    testing = f.readlines()
    f.close()

    testing_targets = []
    testing_images = []
    for j in range(len(testing)):
      i = remove_invisible(testing[j])
      n = normalize_formula(i)
      t = tokenize_formula(n)
      row = [0,0,0,0];
      for j in range(len(first)):
        if (t[0]==first.get(j)):
          row[j] = 1;
      testing_targets.append(row);

    for i in range(len(testing_targets)):
        testing_images.append(str(i))


    # initialize model
    c = img2latexFirst(targs=training_targets, imgs=training_images, test_split=0.1, learning_rate=0.01)
    # process data
    c.prepare_data()
    
    # train model
    c.train_model(
        model_name="m",
        epochs=10,
        batch_size=25
    )
    
    correct = 0;
    # print prediction for each testing example
    for i in range(len(testing_images)):
        r, f, a = c.predict_formula(testing_images[i],testing_targets[i])
        print("predict: ", r)
        print("correct formula: ", f)
        print("did it predict correctly?: ", a)
        if(a=="yes"):
          correct = correct+1
        print("\n")

    print("Percent classified correctly: ", correct/len(testing_images)*100)

